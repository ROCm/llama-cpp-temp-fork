#!/usr/bin/env python3

import argparse
import json
import pathlib
import re
import sys
from typing import Dict, Iterable, List, Tuple


DEFAULT_KERNEL_FAMILY = "qwen3_moe"

SOURCE_ARRAY_TEMPLATE = """static const unsigned char {symbol}[] = {{
{bytes}
}};
static constexpr size_t {symbol}_size = {size};
"""

DEPENDENCY_TABLE_TEMPLATE = """static const kernel_source_span {dependency_table}[] = {{
{dependencies}
}};
"""

SOURCE_RECORD_TEMPLATE = """static const kernel_source {record} = {{
    {{ reinterpret_cast<const char *>({source_symbol}), {source_symbol}_size, KERNEL_SOURCE_FORMAT_TEXT }},
    {dependency_table},
    {dependency_count},
}};
"""

CORPUS_ARRAY_TEMPLATE = """static {type} {symbol}[] = {{
{values}
}};
"""

KERNEL_RECORD_TEMPLATE = """    {{
        {family},
        {name},
        kernel_catalog_id({family}, {name}),
        {source},
        {dependencies},
        {symbol},
        "amdgpu",
        {target_selector},
        {{ nullptr, 0 }},
        {scalar_parameters},
        {bindings},
        {workload_parameters},
        {launch_parameters},
        {{
            {compile_mode},
            {link_module},
            {primary_sources},
            {library_sources},
        }},
    }},"""

SOURCE_DATA_TEMPLATE = """{source_arrays}
{dependency_tables}
{source_records}
static const kernel_source_record_entry kernel_source_records[] = {{
{lookup_entries}
}};
"""

CORPUS_DATA_TEMPLATE = """{kernel_arrays}
{transform_arrays}
static const kernel_definition qwen_kernel_definitions[] = {{
{kernel_records}
}};
{transform_table}

static const kernel_corpus qwen_kernel_corpus = {{
    "ggml-hrx-kernel-corpus-v2",
    {upstream_revision},
    {plan_case_count},
    {{ qwen_kernel_definitions, {kernel_count} }},
    {transform_span},
}};
"""

CATALOG_DATA_TEMPLATE = """struct kernel_catalog_entry {{
    const char * family;
    const char * name;
}};

static constexpr kernel_catalog_entry kernel_catalog_entries[] = {{
{kernel_entries}
}};

constexpr bool kernel_catalog_entry_exists(const char * family, const char * name) {{
    for (const kernel_catalog_entry & known : kernel_catalog_entries) {{
        if (kernel_catalog_name_equal(family, known.family) && kernel_catalog_name_equal(name, known.name)) {{
            return true;
        }}
    }}
    return false;
}}

constexpr bool kernel_catalog_ids_are_unique() {{
    for (size_t i = 0; i < sizeof(kernel_catalog_entries) / sizeof(kernel_catalog_entries[0]); ++i) {{
        for (size_t j = i + 1; j < sizeof(kernel_catalog_entries) / sizeof(kernel_catalog_entries[0]); ++j) {{
            if (kernel_catalog_id(kernel_catalog_entries[i].family, kernel_catalog_entries[i].name) ==
                kernel_catalog_id(kernel_catalog_entries[j].family, kernel_catalog_entries[j].name)) {{
                return false;
            }}
        }}
    }}
    return true;
}}

static_assert(kernel_catalog_ids_are_unique(), "kernel catalog ids must be unique");
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embedded kernel corpus data includes.")
    parser.add_argument("--source-output", type=pathlib.Path, required=True)
    parser.add_argument("--corpus-output", type=pathlib.Path, required=True)
    parser.add_argument("--catalog-output", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--corpus-dir", type=pathlib.Path, required=True)
    parser.add_argument("--depfile", type=pathlib.Path)
    return parser.parse_args()


def qualify_path(prefix: pathlib.Path, value: str) -> str:
    return (prefix / value).as_posix()


def merged_manifest(manifest_paths: List[pathlib.Path], corpus_dir: pathlib.Path) -> dict:
    manifests = []
    for manifest_path in manifest_paths:
        manifest = json.loads(read_text(manifest_path))
        try:
            prefix = manifest_path.resolve().parent.relative_to(corpus_dir.resolve())
        except ValueError as exc:
            raise RuntimeError(f"manifest {manifest_path} is outside corpus directory {corpus_dir}") from exc
        manifest["resources"] = [qualify_path(prefix, path) for path in manifest.get("resources", [])]
        for export in manifest.get("exports", []):
            export["source"] = qualify_path(prefix, export["source"])
            export["compile_dependencies"] = [
                qualify_path(prefix, path) for path in export.get("compile_dependencies", [])
            ]
            recipe = export.get("compile_recipe", {})
            recipe["primary_sources"] = [
                qualify_path(prefix, path) for path in recipe.get("primary_sources", [])
            ]
            recipe["library_sources"] = [
                qualify_path(prefix, path) for path in recipe.get("library_sources", [])
            ]
        manifests.append(manifest)

    exports = {}
    plan_cases = []
    resources = set()
    storage_transforms = {}
    for manifest in manifests:
        for export in manifest.get("exports", []):
            key = (
                export.get("family", DEFAULT_KERNEL_FAMILY),
                export["name"],
                export.get("target_selector", ""),
            )
            if key in exports:
                raise RuntimeError(f"duplicate kernel export {key[0]}:{key[1]}:{key[2]}")
            exports[key] = export
        plan_cases.extend(manifest.get("plan_cases", []))
        resources.update(manifest.get("resources", []))
        for transform in manifest.get("storage_transforms", []):
            match = transform.get("match", {})
            name_match = match.get("name", {})
            key = (
                transform["name"],
                transform.get("target_selector", ""),
                match.get("type", ""),
                tuple(match.get("shape", [])),
                match.get("contiguous", False),
                name_match.get("prefix", ""),
                name_match.get("middle", ""),
                name_match.get("suffix", ""),
            )
            if key in storage_transforms:
                raise RuntimeError(f"duplicate storage transform {key[0]}:{key[1]}:{key[2]}:{key[3]}")
            storage_transforms[key] = transform

    return {
        "upstream_revision": "+".join(manifest.get("upstream_revision", "unknown") for manifest in manifests),
        "exports": [exports[key] for key in sorted(exports)],
        "plan_cases": plan_cases,
        "resources": sorted(resources),
        "storage_transforms": [storage_transforms[key] for key in sorted(storage_transforms)],
    }


def read_text(path: pathlib.Path) -> str:
    try:
        return path.read_text()
    except OSError as exc:
        raise RuntimeError(f"failed to read {path}: {exc}") from exc


def read_bytes(path: pathlib.Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"failed to read {path}: {exc}") from exc


def sanitize_symbol(path: str, index: int) -> str:
    stem = re.sub(r"[^0-9A-Za-z_]", "_", path)
    if not stem or stem[0].isdigit():
        stem = f"_{stem}"
    return f"kernel_source_{index}_{stem}"


def format_byte_array(data: bytes) -> str:
    if not data:
        return ""
    lines = []
    for offset in range(0, len(data), 16):
        chunk = data[offset : offset + 16]
        lines.append("    " + ", ".join(f"0x{byte:02x}" for byte in chunk) + ",")
    return "\n".join(lines) + "\n"


def escape_cpp_string(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def cpp_string(text: str) -> str:
    return f"\"{escape_cpp_string(text)}\""


def resource_access_value(access: str) -> str:
    if access == "read":
        return "ResourceAccess::Read"
    if access == "write":
        return "ResourceAccess::Write"
    if access == "read_write":
        return "ResourceAccess::ReadWrite"
    raise RuntimeError(f"invalid kernel binding access metadata: {access}")


def span_initializer(symbol: str, count: int) -> str:
    if count == 0:
        return "{ nullptr, 0 }"
    return "{ " + symbol + ", " + str(count) + " }"


def typed_array(symbol: str, value_type: str, values: List[str]) -> Tuple[str, str]:
    if not values:
        return "", "{ nullptr, 0 }"
    array = CORPUS_ARRAY_TEMPLATE.format(
        type=value_type,
        symbol=symbol,
        values="\n".join("    " + value + "," for value in values),
    )
    return array, span_initializer(symbol, len(values))


def string_array(symbol: str, items: Iterable[str]) -> Tuple[str, str]:
    return typed_array(symbol, "const char * const", [cpp_string(item) for item in items])


def source_ref_array(symbol: str, items: Iterable[str], source_records: Dict[str, str]) -> Tuple[str, str]:
    values = [
        "{ " + cpp_string(item) + ", &" + source_records[item] + " }"
        for item in items
    ]
    return typed_array(symbol, "const kernel_source_ref", values)


def scalar_array(symbol: str, items: Iterable[dict]) -> Tuple[str, str]:
    values = [
        "{ " + cpp_string(item["name"]) + ", " + cpp_string(item["type"]) + " }"
        for item in items
    ]
    return typed_array(symbol, "const kernel_scalar_definition", values)


def binding_array(symbol: str, names: List[str], access: List[str]) -> Tuple[str, str]:
    if len(names) != len(access):
        raise RuntimeError("kernel binding access metadata has the wrong arity")
    values = [
        "{ " + cpp_string(name) + ", " + resource_access_value(access_value) + " }"
        for name, access_value in zip(names, access)
    ]
    return typed_array(symbol, "const kernel_binding_definition", values)


def scalar_parameter_names(workload_parameters: List[dict], launch_parameters: List[dict]) -> List[str]:
    names: List[str] = []
    for parameter in [*workload_parameters, *launch_parameters]:
        name = parameter["name"]
        if name not in names:
            names.append(name)
    return names


def depfile_escape(path: pathlib.Path) -> str:
    return str(path).replace("\\", "\\\\").replace(" ", "\\ ")


def collect_sources(manifest: dict) -> Tuple[List[str], Dict[str, List[str]]]:
    source_dependencies: Dict[str, List[str]] = {}
    all_sources = set()

    for export in manifest.get("exports", []):
        recipe = export.get("compile_recipe", {})
        primary_sources = recipe.get("primary_sources", [])
        library_sources = recipe.get("library_sources", [])
        if len(primary_sources) != 1:
            raise RuntimeError("expected each export compile_recipe to have exactly one primary source")

        primary = primary_sources[0]
        dependencies = list(library_sources)
        previous = source_dependencies.get(primary)
        if previous is not None and previous != dependencies:
            raise RuntimeError(f"conflicting dependency list for {primary}")
        source_dependencies[primary] = dependencies
        all_sources.add(primary)
        all_sources.update(dependencies)

    all_sources.update(manifest.get("resources", []))

    return sorted(all_sources), source_dependencies


def generate_corpus_records(manifest: dict, source_records: Dict[str, str]) -> Tuple[str, str, int]:
    arrays = []
    records = []
    exports = manifest.get("exports", [])
    for index, export in enumerate(exports):
        recipe = export["compile_recipe"]
        dependencies = list(export["compile_dependencies"])
        library_sources = list(recipe["library_sources"])
        if dependencies != library_sources:
            raise RuntimeError("legacy dependency closure disagrees with compile recipe")
        workload_parameters = list(export["workload_parameters"])
        launch_parameters = list(export["launch_parameters"])

        dependencies_array, dependencies_span = string_array(f"kernel_dependencies_{index}", dependencies)
        scalar_array_text, scalar_span = string_array(
            f"kernel_scalar_parameters_{index}",
            scalar_parameter_names(workload_parameters, launch_parameters),
        )
        bindings_array, bindings_span = binding_array(
            f"kernel_bindings_{index}",
            list(export["bindings"]),
            list(export.get("binding_access", [])),
        )
        workload_array, workload_span = scalar_array(f"kernel_workload_parameters_{index}", workload_parameters)
        launch_array, launch_span = scalar_array(f"kernel_launch_parameters_{index}", launch_parameters)
        primary_array, primary_span = source_ref_array(f"kernel_primary_sources_{index}", recipe["primary_sources"], source_records)
        library_array, library_span = source_ref_array(f"kernel_library_sources_{index}", library_sources, source_records)
        arrays.extend(
            item for item in [
                dependencies_array,
                scalar_array_text,
                bindings_array,
                workload_array,
                launch_array,
                primary_array,
                library_array,
            ] if item
        )
        records.append(
            KERNEL_RECORD_TEMPLATE.format(
                family=cpp_string(export.get("family", DEFAULT_KERNEL_FAMILY)),
                name=cpp_string(export["name"]),
                symbol=cpp_string(export["symbol"]),
                target_selector=cpp_string(export.get("target_selector", "")),
                source=cpp_string(export["source"]),
                dependencies=dependencies_span,
                scalar_parameters=scalar_span,
                bindings=bindings_span,
                workload_parameters=workload_span,
                launch_parameters=launch_span,
                compile_mode=cpp_string(recipe["mode"]),
                link_module=cpp_string(recipe.get("link_module", "")),
                primary_sources=primary_span,
                library_sources=library_span,
            )
        )
    return "\n".join(arrays), "\n".join(records), len(exports)


def storage_transform_kind(value: str) -> str:
    if value == "row_group_field_interleave":
        return "kernel_storage_transform_kind::RowGroupFieldInterleave"
    if value == "row_group_block_group_header_payload":
        return "kernel_storage_transform_kind::RowGroupBlockGroupHeaderPayload"
    raise RuntimeError(f"unsupported storage transform kind: {value}")


def storage_transform_type(value: str) -> str:
    if re.fullmatch(r"[A-Z][A-Z0-9_]*", value) is None:
        raise RuntimeError(f"unsupported storage transform type: {value}")
    return f"GGML_TYPE_{value}"


def generate_storage_transforms(manifest: dict) -> Tuple[str, str, str]:
    arrays = []
    records = []
    transforms = manifest.get("storage_transforms", [])
    for index, encoded in enumerate(transforms):
        shape = list(encoded["match"]["shape"])
        transform = encoded["transform"]
        field_order = list(transform.get("field_order", []))
        if len(shape) != 4 or (transform["kind"] == "row_group_field_interleave" and not field_order):
            raise RuntimeError(f"invalid storage transform {encoded['name']}")
        if field_order:
            field_array, field_span = typed_array(
                f"storage_transform_field_order_{index}",
                "const uint32_t",
                [str(value) for value in field_order],
            )
            arrays.append(field_array)
        else:
            field_span = "{ nullptr, 0 }"
        match = encoded["match"]
        name_match = match.get("name", {})
        records.append(
            "    { "
            + ", ".join([
                cpp_string(encoded["name"]),
                cpp_string(encoded.get("target_selector", "")),
                storage_transform_type(match["type"]),
                "{ " + ", ".join(str(value) for value in shape) + " }",
                "true" if match.get("contiguous", False) else "false",
                cpp_string(name_match.get("prefix", "")),
                cpp_string(name_match.get("suffix", "")),
                "true" if name_match.get("middle") == "decimal" else "false",
                storage_transform_kind(transform["kind"]),
                str(transform["outer_count"]),
                str(transform["row_count"]),
                str(transform["block_count"]),
                str(transform["field_count"]),
                str(transform["unit_bytes"]),
                str(transform["row_group"]),
                str(transform.get("block_group", 0)),
                str(transform.get("header_fields", 0)),
                field_span,
            ])
            + " },"
        )
    if not records:
        return "", "", "{ nullptr, 0 }"
    table = "static const kernel_storage_transform qwen_storage_transforms[] = {\n" + "\n".join(records) + "\n};"
    return "\n".join(arrays), table, "{ qwen_storage_transforms, " + str(len(records)) + " }"


def generate_catalog_verifier(manifest: dict) -> str:
    kernel_entries = sorted(set(
        (export.get("family", DEFAULT_KERNEL_FAMILY), export["name"])
        for export in manifest.get("exports", [])
    ))
    return CATALOG_DATA_TEMPLATE.format(
        kernel_entries="\n".join(
            "    { " + cpp_string(family) + ", " + cpp_string(kernel_name) + " },"
            for family, kernel_name in kernel_entries
        ),
    )


def generate_includes(args: argparse.Namespace, manifest: dict) -> Tuple[str, str, str, List[pathlib.Path], int]:
    corpus_dir = args.corpus_dir
    sources, source_dependencies = collect_sources(manifest)
    source_bytes: Dict[str, bytes] = {}
    input_files: List[pathlib.Path] = []

    for export in manifest.get("exports", []):
        recipe = export.get("compile_recipe", {})
        for primary in recipe.get("primary_sources", []):
            if primary not in sources:
                raise RuntimeError(f"export primary source is not embedded: {primary}")

    for source in sources:
        path = corpus_dir / source
        input_files.append(path)
        source_bytes[source] = read_bytes(path)

    source_symbols: Dict[str, str] = {}
    source_arrays = []
    for index, source in enumerate(sources):
        symbol = sanitize_symbol(source, index)
        source_symbols[source] = symbol
        data = source_bytes[source]
        source_arrays.append(
            SOURCE_ARRAY_TEMPLATE.format(
                symbol=symbol,
                bytes=format_byte_array(data),
                size=len(data),
            )
        )

    dependency_tables = []
    source_record_definitions = []
    source_record_symbols = {}
    lookup_entries = []
    for index, source in enumerate(sources):
        dependencies = source_dependencies.get(source, [])
        record = f"kernel_source_record_{index}"
        source_record_symbols[source] = record
        if dependencies:
            dependency_table = f"kernel_source_dependencies_{index}"
            entries = []
            for dependency in dependencies:
                symbol = source_symbols[dependency]
                entries.append(
                    f"    {{ reinterpret_cast<const char *>({symbol}), {symbol}_size, KERNEL_SOURCE_FORMAT_TEXT }},"
                )
            dependency_tables.append(
                DEPENDENCY_TABLE_TEMPLATE.format(
                    dependency_table=dependency_table,
                    dependencies="\n".join(entries),
                )
            )
        else:
            dependency_table = "nullptr"

        source_record_definitions.append(
            SOURCE_RECORD_TEMPLATE.format(
                record=record,
                source_symbol=source_symbols[source],
                dependency_table=dependency_table,
                dependency_count=len(dependencies),
            )
        )
        lookup_entries.append(f"    {{ {cpp_string(source)}, &{record} }},")

    kernel_arrays, kernel_records, kernel_count = generate_corpus_records(manifest, source_record_symbols)
    transform_arrays, transform_table, transform_span = generate_storage_transforms(manifest)
    return (
        SOURCE_DATA_TEMPLATE.format(
            source_arrays="\n".join(source_arrays),
            dependency_tables="\n".join(dependency_tables),
            source_records="\n".join(source_record_definitions),
            lookup_entries="\n".join(lookup_entries),
        ),
        CORPUS_DATA_TEMPLATE.format(
            kernel_arrays=kernel_arrays,
            transform_arrays=transform_arrays,
            kernel_records=kernel_records,
            transform_table=transform_table,
            transform_span=transform_span,
            upstream_revision=cpp_string(manifest["upstream_revision"]),
            plan_case_count=len(manifest["plan_cases"]),
            kernel_count=kernel_count,
        ),
        generate_catalog_verifier(manifest),
        input_files,
        sum(len(data) for data in source_bytes.values()),
    )


def write_depfile(path: pathlib.Path, outputs: List[pathlib.Path], inputs: List[pathlib.Path]) -> None:
    targets = " ".join(depfile_escape(output_path) for output_path in outputs)
    entries = [depfile_escape(input_path) for input_path in inputs]
    path.write_text(f"{targets}: {' '.join(entries)}\n")


def main() -> int:
    args = parse_args()
    try:
        manifest = merged_manifest(args.manifest, args.corpus_dir)
        source_include, corpus_include, catalog_include, input_files, byte_count = generate_includes(args, manifest)
        args.source_output.parent.mkdir(parents=True, exist_ok=True)
        args.corpus_output.parent.mkdir(parents=True, exist_ok=True)
        args.catalog_output.parent.mkdir(parents=True, exist_ok=True)
        args.source_output.write_text(source_include)
        args.corpus_output.write_text(corpus_include)
        args.catalog_output.write_text(catalog_include)
        if args.depfile is not None:
            args.depfile.parent.mkdir(parents=True, exist_ok=True)
            write_depfile(args.depfile, [args.source_output, args.corpus_output, args.catalog_output],
                          [*args.manifest, *input_files])
    except Exception as exc:
        print(f"generate_kernel_corpus.py: {exc}", file=sys.stderr)
        return 1

    print(f"embedded kernel corpus: source_files={len(input_files)} source_bytes={byte_count}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
