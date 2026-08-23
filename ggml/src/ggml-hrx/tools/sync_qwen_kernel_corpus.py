#!/usr/bin/env python3
"""Mirrors the bounded Qwen MoE Loom corpus with reproducible provenance."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile


SOURCE_SUBDIR = pathlib.Path("experimental/qwen_moe/kernels")
QWEN_ENDPOINT_SOURCE_SUBDIR = pathlib.Path("experimental/qwen/kernels")
CORPUS_FILES = (
    "ggml/linear_q6k_f32.loom",
    "ggml/linear_q6k_q8_1_x4.loom",
    "ggml/quantize_q8_1_x4.loom",
    "qwen3_moe/attention_qkv_postprocess_fused.loom",
    "qwen3_moe/attention_qkv_quantized.loom",
    "qwen3_moe/attention_qkv_same_format_prefill.loom",
    "qwen3_moe/batched_decode_expert_dispatch.loom",
    "qwen3_moe/batched_decode_gate_up_q4k.loom",
    "qwen3_moe/expert_table_partition_fused.loom",
    "qwen3_moe/flash_attention_decode_f32_f16_wmma.loom",
    "qwen3_moe/flash_attention_decode_q128_f32_f16_wmma.loom",
    "qwen3_moe/flash_attention_decode_split_f32_f16_wmma.loom",
    "qwen3_moe/flash_attention_decode_split_next_q8_test.loom",
    "qwen3_moe/model_config.loom",
    "qwen3_moe/routed_down_q4k.loom",
    "qwen3_moe/routed_down_q6k.loom",
    "qwen3_moe/routed_down_next_q8.loom",
    "qwen3_moe/routed_down_quantized_f16_wmma.loom",
    "qwen3_moe/routed_down_weighted_reduce_next_rmsnorm_f32.loom",
    "qwen3_moe/routed_down_weighted_reduce_next_rmsnorm_q8_1_x4.loom",
    "qwen3_moe/routed_gate_up_swiglu_q4k.loom",
    "qwen3_moe/routed_linear_q4k_f16_wmma.loom",
    "qwen3_moe/router_projection_f32.loom",
    "qwen3_moe/router_projection_top8_fused_f32.loom",
    "qwen3_moe/router_top8_f32.loom",
)
QWEN_ENDPOINT_FILES = (
    ("token_embedding_q4k.loom", "qwen_owned/token_embedding_q4k.loom"),
    ("attention_metadata.loom", "qwen_owned/attention_metadata.loom"),
)

# These integration kernels are deliberately owned by the llama.cpp HRX
# backend. They are not attributed to the pinned qwen_moe corpus or its BUILD
# recipes.
OWNED_KERNEL_DIR = pathlib.Path(__file__).resolve().parent.parent / "kernel-corpus" / "kernels"
OWNED_CORPUS_FILES = (
    "qwen3_moe/attention_postprocess_f32_f16.loom",
    "qwen3_moe/attention_prepare_quantized.loom",
    "qwen3_moe/dense_linear_quantized_f16_wmma.loom",
    "qwen3_moe/flash_attention_f32_f16_wmma.loom",
)
OWNED_FILES = (
    "qwen_owned/token_embedding_bringup_workaround.loom",
    "qwen_owned/attention_state_initialize.loom",
    "qwen_owned/attention_metadata_bringup_workaround.loom",
    "qwen_owned/concat_window_tail_f32.loom",
    "qwen_owned/gated_delta_net_f32.loom",
    "qwen_owned/ssm_conv_f32.loom",
    "hrx_owned/gather_add_f32.loom",
    "hrx_owned/add_f32.loom",
    "hrx_owned/copy_f32.loom",
)

KERNEL_RE = re.compile(
    r"kernel\.def(?P<modifiers>(?:\s+(?:target\([^)]*\)|export\(\"[^\"]+\"\)))*)"
    r"\s+@(?P<symbol>[A-Za-z0-9_]+)"
    r"\((?P<workload>.*?)\)\s*\{.*?\}\s*launch\((?P<launch>.*?)\)"
    r"(?:\s+where\s+\[[^\]]*\])?\s*\{",
    re.DOTALL,
)
ARG_RE = re.compile(r"%(?P<name>[A-Za-z0-9_]+)\s*:\s*(?P<type>[A-Za-z0-9<>?]+)")
TARGET_MODIFIER_RE = re.compile(r"target\(@(?P<symbol>[A-Za-z0-9_]+)\)")
EXPORT_MODIFIER_RE = re.compile(r"export\(\"(?P<name>[^\"]+)\"\)")
AMDGPU_TARGET_RE = re.compile(
    r"amdgpu\.target<(?P<selector>[A-Za-z0-9_.-]+)>\s+@(?P<symbol>[A-Za-z0-9_]+)"
)
LEGACY_TEMPLATE_USE_RE = re.compile(r"func\.(?:apply|ukernel)<([^>]+)>")
LEGACY_TEMPLATE_DEF_RE = re.compile(
    r"^(?P<indent>\s*)func\.template<(?P<family>[^>]+)>(?P<tail>.*)$")
LEGACY_TEMPLATE_APPLY_RE = re.compile(
    r"func\.apply<(?P<family>[^>]+)>\((?P<args>[^)]*)\)")
LEGACY_INLINE_CALL_RE = re.compile(
    r"func\.call inline @(?P<provider>[A-Za-z0-9_.$-]+)")


def binding_access(symbol: str, name: str) -> str:
    """Authoritative launch ABI access contract; no name inference at runtime."""
    if symbol == "qwen3_quant_act_u4asym":
        return "write" if name in ("qs", "ds", "meta") else "read"
    if symbol == "qwen3_quant_act_i4":
        return "write" if name in ("qs", "ds", "sums") else "read"
    if symbol in ("qwen3_moe_dense_linear_q4k_u4asym_prepacked_wmmai4_64x128x64",
                  "qwen3_moe_dense_linear_q4k_u4asym_prepacked_wmmai4_64x16x64"):
        return "write" if name == "dst" else "read"
    if symbol == "qwen3_moe_dense_linear_q4k_u4asym_prepacked_wmmai4_64x16x64_splitk2":
        return "read_write" if name in ("dst", "partial", "completion_counters") else "read"
    if symbol in ("qwen3_moe_dense_linear_q4k_u4asym_prepacked_dual_grid_64x16x64",
                  "qwen3_moe_dense_linear_symi4_i4_adjacent_dual_grid_m16n16_wg64"):
        return "write" if name in ("gate_output", "up_output") else "read"
    if symbol in ("qwen3_moe_dense_linear_symi4_i4_adjacent_m16n16_wg64",
                  "qwen3_moe_dense_linear_symi2_i4_adjacent_m16n16_wg64"):
        return "write" if name == "output" else "read"
    if symbol == "qwen3_moe_dense_linear_symi4_i4_adjacent_m16n16_wg64_splitk2":
        return "read_write" if name in ("output", "partial", "completion_counters") else "read"
    if symbol == "ggml_top_k64_f32_partitions_register":
        return "write" if name in ("partial_values", "partial_ids") else "read"
    if symbol == "ggml_top_k64_f32_reduce_gather_register":
        return "write" if name in ("candidate_output", "logits_output") else "read"
    if symbol == "qwen3_moe_dense_q6k_packed_raw_selected_refine_64x16":
        return "write" if name in ("exact_output", "logits") else "read"
    if symbol == "qwen3_moe_fill_negative_f32":
        return "write" if name == "output" else "read"
    if symbol == "qwen3_moe_dense_linear_q4k_i4_dual_gate_up_swiglu_m32n32_f16out":
        return "write" if name == "dst" else "read"
    if symbol == "qwen3_moe_dense_linear_q4k_i4_dual_gate_up_swiglu_m32n32_u4out":
        return "write" if name in ("dst", "qout_qs", "qout_ds", "qout_sums") else "read"
    if symbol == "qwen3_moe_dense_linear_symi4_i4_dual_gate_up_swiglu_m32n32_f32out":
        return "write" if name == "dst" else "read"
    if symbol == "hrx2_concat_window_tail":
        return "write" if name in ("dst", "cache") else "read"
    if symbol in ("hrx2_ssm_conv_f32_state_materialized_decode_silu",
                  "hrx2_ssm_conv_f32_chan_concat_silu_regblock_wg1024"):
        return "write" if name in ("dst", "cache") else "read"
    if symbol == "hrx2_ssm_conv_f32_state_materialized_rollback_silu":
        return "write" if name == "dst" or name.startswith("cache") else "read"
    if symbol == "copy_f32_f32_contiguous_1d":
        return "write" if name == "dst" else "read"
    if symbol == "hrx2_gdn_projection_epilogue_f32":
        return "write" if name in ("gate_dst", "beta_dst") else "read"
    if symbol in ("hrx2_recurrent_rms_raw_gate_silu_mul_f32",
                  "hrx2_recurrent_rms_raw_gate_silu_mul_f32_f16",
                  "hrx2_recurrent_rms_raw_gate_silu_mul_f32_i4"):
        return "write" if name in ("dst", "f16_dst", "i4_qs", "i4_ds", "i4_sums") else "read"
    if symbol in (
        "hrx2_swiglu_split_f32",
        "hrx2_swiglu_split_f32_f16",
        "hrx2_swiglu_split_f32_i4_parallel",
    ):
        return "write" if name in ("dst", "f16_dst", "i4_qs", "i4_ds", "i4_sums") else "read"
    if symbol in ("hrx2_rms_norm_strided_mul_rope_f32", "hrx2_set_rows_f32_f16"):
        return "write" if name == "dst" else "read"
    if symbol == "hrx2_flash_attn_ext_f32_f16_direct_kv64_f32acc_gate":
        return "write" if name == "output" else "read"
    if symbol == "hrx2_gated_delta_net_f32_sv128_qk_l2_full_head_rms_scale_fixed":
        return "write" if name in ("dst", "rms_scales") else "read"
    if symbol == "hrx2_gated_delta_net_f32_sv128_qk_l2_full_head_rms_scale_state_cache_fixed":
        if name == "state_inout":
            return "read_write"
        return "write" if name in ("dst", "rms_scales") else "read"
    if symbol == "hrx2_gated_delta_net_f32_sv128_qk_l2_full_head_rms_scale_snapshot_rollback_fixed":
        return "write" if name in ("snapshot_cache", "dst", "rms_scales") else "read"
    if symbol == "qwen_attention_context_base_capture":
        return "read" if name == "positions" else "write"
    if symbol == "qwen_attention_decode_state_initialize":
        return "read" if name == "positions" else "write"
    if symbol == "qwen_attention_metadata_bringup_workaround" and name != "control":
        return "read_write"
    if symbol == "qwen_attention_metadata" and name != "control":
        return "read_write"
    if symbol == "qwen_decode_attention_metadata" and name != "control":
        return "read_write"
    if symbol == "qwen3_moe_router_top8_f32" and name in ("route_ids", "route_weights"):
        return "write"
    if symbol == "qwen3_moe_router_projection_top8_fused_decode_f32" and name in (
            "logits", "completion_counter", "route_ids", "route_weights"):
        return "read_write"
    if symbol == "qwen3_moe_attention_qkv_postprocess_fused_decode" and name in (
            "query_output_raw", "key_output_raw", "value_output_raw",
            "query_output", "key_cache", "value_cache", "completion_counters"):
        return "read_write"
    if symbol == "qwen3_moe_build_expert_table" and name == "expert_table":
        return "write"
    if symbol == "qwen3_moe_build_expert_partition_table" and name == "partition_table":
        return "write"
    if symbol == "qwen3_moe_build_expert_table_partition_prefill_512" and name in ("expert_table", "partition_table"):
        return "write"
    if symbol == "qwen3_moe_routed_down_weighted_reduce_next_rmsnorm_f32":
        if name == "hidden_state":
            return "read_write"
        if name == "next_projection_input":
            return "write"
    if symbol == "qwen3_moe_rmsnorm_f32_f16" and name == "f16_output":
        return "write"
    if symbol == "qwen3_moe_rmsnorm_f32_i4" and name in ("i4_qs", "i4_ds", "i4_sums"):
        return "write"
    if symbol == "qwen3_moe_add_rmsnorm_f32_i4" and name in (
            "residual_output", "output", "i4_qs", "i4_ds", "i4_sums"):
        return "write"
    if symbol == "ggml_q8_1_x4_inspect_one_group" and name != "packed":
        return "write"
    if name in ("output", "query_output", "key_output", "value_output", "normalized_output", "q8_output",
                "next_q8_output", "key_cache", "value_cache", "partial_max", "partial_sum", "partial_output",
                "completion_counter", "completion_counters"):
        return "read_write"
    return "read"


def starlark_calls(text: str, function: str) -> list[str]:
    """Extracts the literal-only calls used by the pinned kernel BUILD file."""
    result: list[str] = []
    marker = function + "("
    cursor = 0
    while (start := text.find(marker, cursor)) != -1:
        index = start + len(marker)
        depth = 1
        quote: str | None = None
        escaped = False
        while index < len(text) and depth:
            character = text[index]
            if quote:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
            elif character in "\"'":
                quote = character
            elif character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
            index += 1
        if depth:
            raise RuntimeError(f"unterminated {function} call in BUILD.bazel")
        result.append(text[start + len(marker) : index - 1])
        cursor = index
    return result


def literal_assignment(call: str, name: str, default: object = None) -> object:
    match = re.search(rf"(?:^|\n)\s*{re.escape(name)}\s*=\s*(\[[\s\S]*?\]|\"[^\"]*\")\s*,", call)
    return default if match is None else ast.literal_eval(match.group(1))


def parse_build_recipes(text: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    modules: list[dict[str, object]] = []
    for call in starlark_calls(text, "loom_link_module"):
        modules.append({
            "name": literal_assignment(call, "name"),
            "srcs": literal_assignment(call, "srcs", []),
            "libraries": literal_assignment(call, "libraries", []),
        })
    module_names = {str(module["name"]) for module in modules}
    cases: list[dict[str, object]] = []
    for call in starlark_calls(text, "iree_executable_test"):
        name = literal_assignment(call, "name")
        if not isinstance(name, str) or not name.endswith("_plan_test"):
            continue
        args = literal_assignment(call, "args", [])
        data = literal_assignment(call, "data", [])
        linked = [item[1:] for item in data if isinstance(item, str) and item.startswith(":") and item[1:] in module_names]
        direct_sources = [item for item in data if isinstance(item, str) and item.endswith(".loom")]
        if len(linked) + len(direct_sources) != 1:
            raise RuntimeError(f"plan test {name} does not name exactly one linked module or direct Loom source")
        case = {
            "name": name,
            "args": args,
        }
        if linked:
            case["link_module"] = linked[0]
        else:
            case["source"] = direct_sources[0]
        cases.append(case)
    if not modules or not cases:
        raise RuntimeError("BUILD.bazel contains no pinned Loom link/plan recipes")
    return modules, cases


def git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def optional_git(repo: pathlib.Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def normalize_repository_url(url: str) -> str:
    """Normalizes GitHub transport spelling without changing repository identity."""
    match = re.fullmatch(r"(?:ssh://)?git@github\.com[:/](?P<path>.+)", url)
    if match:
        return f"https://github.com/{match.group('path')}"
    return url


def upstream_repository(repo: pathlib.Path) -> str:
    """Returns provenance without imposing a local Git remote name."""
    remotes: list[str] = []
    branch = optional_git(repo, "symbolic-ref", "--quiet", "--short", "HEAD")
    if branch:
        branch_remote = optional_git(repo, "config", "--get", f"branch.{branch}.remote")
        if branch_remote and branch_remote != ".":
            remotes.append(branch_remote)
    remotes.append("origin")
    remotes.extend(git(repo, "remote").splitlines())
    for remote in dict.fromkeys(remotes):
        url = optional_git(repo, "config", "--get", f"remote.{remote}.url")
        if url:
            return normalize_repository_url(url)
    raise RuntimeError("HRX source tree has no repository remote for provenance")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def template_provider_contract(line: str) -> tuple[str, str, str]:
    match = LEGACY_TEMPLATE_DEF_RE.match(line)
    if not match:
        raise ValueError(f"not a legacy template definition: {line!r}")
    tail = match.group("tail")
    providers = list(re.finditer(r"@(?P<symbol>[A-Za-z0-9_.$-]+)\s*(?=\()", tail))
    provider = providers[-1] if providers else None
    signature_start = tail.find("(", provider.start()) if provider else -1
    body_start = tail.rfind("{")
    if provider is None or signature_start < 0 or body_start < signature_start:
        raise ValueError(f"unsupported template header: {line!r}")
    signature = tail[signature_start:body_start].strip()
    signature = re.sub(r"\s+where\s+\[.*\]\s*$", "", signature)
    return match.group("family"), provider.group("symbol"), signature


def normalized_signature(signature: str) -> str:
    return re.sub(r"%[A-Za-z0-9_.$-]+\s*:\s*", "", re.sub(r"\s+", "", signature))


def template_family_contracts(
        *directories: pathlib.Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], set[str]]:
    signatures: dict[str, str] = {}
    providers: dict[str, str] = {}
    families_to_providers: dict[str, list[str]] = {}
    host_families: dict[str, bool] = {}
    for directory in directories:
        for path in sorted(directory.rglob("*.loom")):
            for line in path.read_text().splitlines():
                match = LEGACY_TEMPLATE_DEF_RE.match(line)
                if not match:
                    continue
                family, provider, signature = template_provider_contract(line)
                prior_signature = signatures.get(family)
                if (prior_signature is not None and
                        normalized_signature(prior_signature) != normalized_signature(signature)):
                    raise RuntimeError(f"template family {family} has incompatible contracts")
                prior_family = providers.get(provider)
                if prior_family is not None and prior_family != family:
                    raise RuntimeError(f"template provider {provider} has incompatible families")
                signatures.setdefault(family, signature)
                providers.setdefault(provider, family)
                families_to_providers.setdefault(family, []).append(provider)
                is_device = re.search(r"(?:^|\s)device(?:\s|$)", match.group("tail")) is not None
                host_families[family] = host_families.get(family, True) and not is_device

    launch_providers = {
        family: family_providers[0]
        for family, family_providers in families_to_providers.items()
        if family.endswith(".launch") and len(family_providers) == 1 and host_families[family]
    }
    pure_template_families = {
        family
        for family, signature in signatures.items()
        if (family not in launch_providers and host_families.get(family, False) and
            normalized_signature(signature).startswith("()"))
    }
    return signatures, providers, launch_providers, pure_template_families


def code_brace_delta(line: str) -> int:
    code = line.split("//", 1)[0]
    return code.count("{") - code.count("}")


def migrate_template_families(
        data: bytes, source: str, signatures: dict[str, str], providers: dict[str, str],
        launch_providers: dict[str, str], pure_template_families: set[str]) -> bytes:
    """Migrates the pinned pre-symbol template syntax to the current Loom form."""
    original = data.decode("utf-8")
    exact_inline_providers = {
        match.group("provider") for match in LEGACY_INLINE_CALL_RE.finditer(original)
        if match.group("provider") in providers
    }
    inline_families = {
        providers[match.group("provider")]
        for match in LEGACY_INLINE_CALL_RE.finditer(original)
        if match.group("provider") in providers and match.group("provider") not in exact_inline_providers
    }
    provider_families = set()
    for line in original.splitlines():
        header = LEGACY_TEMPLATE_DEF_RE.match(line)
        if header:
            family, provider, _ = template_provider_contract(line)
            if provider not in exact_inline_providers:
                provider_families.add(family)
    families = sorted(set(LEGACY_TEMPLATE_USE_RE.findall(original)) | provider_families | inline_families)
    if not families and not exact_inline_providers:
        return data
    missing = [family for family in families if family not in signatures]
    if missing:
        raise RuntimeError(f"{source}: missing template provider contracts for {missing}")

    transformed: list[str] = []
    template_depth = 0
    in_template = False
    return_op = "template.return"

    def migrate_apply(match: re.Match[str]) -> str:
        family = match.group("family")
        args = match.group("args")
        if family in launch_providers:
            return f"func.call pure inline @{launch_providers[family]}({args})"
        purity = " pure" if family in pure_template_families else ""
        return f"template.apply<@{family}>({args}){purity}"

    def migrate_inline_call(match: re.Match[str]) -> str:
        if match.group("provider") in exact_inline_providers:
            return match.group(0)
        family = providers.get(match.group("provider"))
        if family is None:
            return match.group(0)
        if family in launch_providers:
            return "func.call pure inline @" + match.group("provider")
        return "template.apply<@" + family + ">"

    for line in original.splitlines(keepends=True):
        header = LEGACY_TEMPLATE_DEF_RE.match(line.rstrip("\n"))
        if header:
            family, provider, _ = template_provider_contract(line.rstrip("\n"))
            if provider in exact_inline_providers:
                replacement = "func.def inline"
                return_op = "func.return"
            elif family in launch_providers:
                replacement = "func.def pure"
                return_op = "func.return"
            else:
                purity = " pure" if family in pure_template_families else ""
                replacement = f"template.def<@{family}>{purity}"
                return_op = "template.return"
            line = line.replace(f"func.template<{family}>", replacement, 1)
            if provider in exact_inline_providers:
                line = line.replace("func.def inline device ", "func.def inline ", 1)
            in_template = True
            template_depth = code_brace_delta(line)
        else:
            line = LEGACY_TEMPLATE_APPLY_RE.sub(migrate_apply, line)
            line = re.sub(r"func\.ukernel<([^>]+)>", r"template.ukernel<@\1>", line)
            line = LEGACY_INLINE_CALL_RE.sub(migrate_inline_call, line)
            if in_template:
                line = line.replace("func.return", return_op)
                template_depth += code_brace_delta(line)
                if template_depth == 0:
                    in_template = False
                elif template_depth < 0:
                    raise RuntimeError(f"{source}: unbalanced template provider body")
        transformed.append(line)
    if in_template:
        raise RuntimeError(f"{source}: unterminated template provider body")

    declarations = "".join(
        f"template.decl{' pure' if family in pure_template_families else ''} "
        f"@{family}{signatures[family]}\n"
        for family in families if family not in launch_providers)
    return (declarations + "\n" + "".join(transformed)).encode("utf-8")


def migrate_fragment_extent_proofs(data: bytes, source: str) -> bytes:
    """Keeps the vector origin related to its proven end after folding."""
    if source != "qwen3_moe/flash_attention_decode_split_f32_f16_wmma.loom":
        return data
    replacements = (
        (
            b"%score_key_origin, %score_key_end = index.assume %score_key_origin0, %score_key_end0 [le(%score_key_end0, %bounded_key_value_token_count)] : index, index",
            b"%score_key_end = index.assume %score_key_end0 [le(%score_key_end0, %bounded_key_value_token_count)] : index\n      %score_key_origin = index.sub %score_key_end, %c16 : index",
        ),
        (
            b"%value_token, %value_token_end = index.assume %value_token0, %value_token_end0 [le(%value_token_end0, %bounded_key_value_token_count)] : index, index",
            b"%value_token_end = index.assume %value_token_end0 [le(%value_token_end0, %bounded_key_value_token_count)] : index\n      %value_token = index.sub %value_token_end, %c16 : index",
        ),
    )
    for old, new in replacements:
        if data.count(old) != 1:
            raise RuntimeError(f"{source}: fragment extent proof does not match the pinned source")
        data = data.replace(old, new)
    return data


def canonicalize_loom(data: bytes, source: str, loom_format: pathlib.Path) -> bytes:
    result = subprocess.run(
        [str(loom_format), "--from=text", "--to=text"], input=data,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise RuntimeError(
            f"loom-format rejected migrated source {source}:\n{result.stderr.decode('utf-8')}")
    return result.stdout


def parse_exports(text: str, source: str) -> list[dict[str, object]]:
    exports: list[dict[str, object]] = []
    for match in KERNEL_RE.finditer(text):
        modifiers = match.group("modifiers")
        target_match = TARGET_MODIFIER_RE.search(modifiers)
        export_match = EXPORT_MODIFIER_RE.search(modifiers)
        workload = [item.groupdict() for item in ARG_RE.finditer(match.group("workload"))]
        launch = [item.groupdict() for item in ARG_RE.finditer(match.group("launch"))]
        bindings = [item["name"] for item in launch if item["type"] == "buffer"]
        exports.append(
            {
                "name": export_match.group("name") if export_match else match.group("symbol"),
                "symbol": match.group("symbol"),
                "target_symbol": target_match.group("symbol") if target_match else "",
                "source": source,
                "workload_parameters": workload,
                "launch_parameters": [item for item in launch if item["type"] != "buffer"],
                "bindings": bindings,
                "binding_access": [binding_access(match.group("symbol"), name) for name in bindings],
            }
        )
    return exports


def parse_amdgpu_targets(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for match in AMDGPU_TARGET_RE.finditer(text):
        symbol = match.group("symbol")
        selector = match.group("selector")
        if symbol in result and result[symbol] != selector:
            raise RuntimeError(
                f"AMDGPU target @{symbol} is declared as both {result[symbol]} and {selector}")
        result[symbol] = selector
    return result


def merge_amdgpu_targets(target_selectors: dict[str, str], additions: dict[str, str]) -> None:
    for symbol, selector in additions.items():
        if symbol in target_selectors and target_selectors[symbol] != selector:
            raise RuntimeError(
                f"AMDGPU target @{symbol} is declared as both {target_selectors[symbol]} and {selector}")
        target_selectors[symbol] = selector


def resolve_export_variants(exports: list[dict[str, object]], target_selectors: dict[str, str]) -> None:
    groups: dict[str, list[dict[str, object]]] = {}
    for item in exports:
        groups.setdefault(str(item["name"]), []).append(item)

    for name, variants in groups.items():
        selectors: set[str] = set()
        for item in variants:
            target_symbol = str(item.pop("target_symbol"))
            selector = ""
            if target_symbol:
                if target_symbol not in target_selectors:
                    raise RuntimeError(f"kernel export {name} references unknown target @{target_symbol}")
                selector = target_selectors[target_symbol]
                if selector.endswith("-generic"):
                    selector = ""
            if selector in selectors:
                label = selector or "default"
                raise RuntimeError(f"kernel export {name} repeats target variant {label}")
            selectors.add(selector)
            item["target_selector"] = selector


def construct(
        source_root: pathlib.Path, destination: pathlib.Path, expected_revision: str | None,
        loom_format: pathlib.Path) -> None:
    revision = git(source_root, "rev-parse", "HEAD")
    if expected_revision and revision != expected_revision:
        raise RuntimeError(f"HRX revision {revision} does not match expected {expected_revision}")
    if git(source_root, "status", "--porcelain"):
        raise RuntimeError("refusing to mirror a dirty HRX source tree")

    source_directory = source_root / SOURCE_SUBDIR
    endpoint_source_directory = source_root / QWEN_ENDPOINT_SOURCE_SUBDIR
    (family_signatures, provider_families, launch_providers,
     pure_template_families) = template_family_contracts(
         source_directory, endpoint_source_directory)

    def read_upstream_source(path: pathlib.Path, logical_name: str) -> bytes:
        migrated = migrate_template_families(
            path.read_bytes(), logical_name, family_signatures, provider_families,
            launch_providers, pure_template_families)
        migrated = migrate_fragment_extent_proofs(migrated, logical_name)
        return canonicalize_loom(migrated, logical_name, loom_format)

    def read_owned_source(path: pathlib.Path, logical_name: str) -> bytes:
        return canonicalize_loom(path.read_bytes(), logical_name, loom_format)

    build_data = (source_directory / "BUILD.bazel").read_bytes()
    all_link_modules, all_plan_cases = parse_build_recipes(build_data.decode("utf-8"))

    modules_by_name = {str(item["name"]): item for item in all_link_modules}
    direct_plan_sources = {
        str(item["source"])
        for item in all_plan_cases
        if "source" in item
    }

    def module_files(name: str) -> list[str]:
        module = modules_by_name[name]
        result = list(module["srcs"])
        for library in module["libraries"]:
            if str(library).startswith(":"):
                result.extend(module_files(str(library)[1:]))
            else:
                result.append(str(library))
        return list(dict.fromkeys(result))

    def compile_recipe(source: str) -> dict[str, object]:
        direct = [str(item["name"]) for item in all_link_modules if source in item["srcs"]]
        indirect = [str(item["name"]) for item in all_link_modules if source in item["libraries"]]
        if not direct and (source in direct_plan_sources or not indirect):
            return {"mode": "direct", "primary_sources": [source], "library_sources": []}
        module_name = (direct or indirect)[0]
        module = modules_by_name[module_name]
        files = module_files(module_name)
        return {
            "mode": "archive",
            "link_module": module_name,
            "primary_sources": list(module["srcs"]),
            "library_sources": [item for item in files if item not in module["srcs"]],
        }
    file_rows: list[dict[str, object]] = []
    exports: list[dict[str, object]] = []
    target_selectors: dict[str, str] = {}
    upstream_aggregate = hashlib.sha256()
    for relative_text in CORPUS_FILES:
        relative = pathlib.Path(relative_text)
        source = source_directory / relative
        if not source.is_file():
            raise RuntimeError(f"missing required corpus source: {source}")
        data = read_upstream_source(source, relative_text)
        digest = sha256(data)
        upstream_aggregate.update(relative_text.encode())
        upstream_aggregate.update(b"\0")
        upstream_aggregate.update(bytes.fromhex(digest))
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        file_rows.append({"path": relative_text, "sha256": digest, "size": len(data)})
        source_text = data.decode("utf-8")
        exports.extend(parse_exports(source_text, relative_text))
        merge_amdgpu_targets(target_selectors, parse_amdgpu_targets(source_text))

    for source_text_name, local_text_name in QWEN_ENDPOINT_FILES:
        source = endpoint_source_directory / source_text_name
        if not source.is_file():
            raise RuntimeError(f"missing required Qwen endpoint source: {source}")
        provenance_path = f"{QWEN_ENDPOINT_SOURCE_SUBDIR.as_posix()}/{source_text_name}"
        data = read_upstream_source(source, provenance_path)
        digest = sha256(data)
        upstream_aggregate.update(provenance_path.encode())
        upstream_aggregate.update(b"\0")
        upstream_aggregate.update(bytes.fromhex(digest))
        relative_text = f"../{local_text_name}"
        target = destination.parent / local_text_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        file_rows.append({
            "path": relative_text,
            "sha256": digest,
            "size": len(data),
            "upstream_path": provenance_path,
        })
        source_text = data.decode("utf-8")
        exports.extend(parse_exports(source_text, relative_text))
        merge_amdgpu_targets(target_selectors, parse_amdgpu_targets(source_text))

    owned_aggregate = hashlib.sha256()
    for filename in OWNED_CORPUS_FILES:
        source = OWNED_KERNEL_DIR / "qwen_moe" / filename
        if not source.is_file():
            raise RuntimeError(f"missing required backend-owned corpus source: {source}")
        data = read_owned_source(source, filename)
        digest = sha256(data)
        target = destination / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        owned_aggregate.update(filename.encode())
        owned_aggregate.update(b"\0")
        owned_aggregate.update(bytes.fromhex(digest))
        file_rows.append({"path": filename, "sha256": digest, "size": len(data), "owner": "ggml-hrx"})
        source_text = data.decode("utf-8")
        exports.extend(parse_exports(source_text, filename))
        merge_amdgpu_targets(target_selectors, parse_amdgpu_targets(source_text))

    for filename in OWNED_FILES:
        source = OWNED_KERNEL_DIR / filename
        if not source.is_file():
            raise RuntimeError(f"missing required backend-owned kernel source: {source}")
        data = read_owned_source(source, filename)
        digest = sha256(data)
        target = destination.parent / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        relative_text = f"../{filename}"
        owned_aggregate.update(filename.encode())
        owned_aggregate.update(b"\0")
        owned_aggregate.update(bytes.fromhex(digest))
        file_rows.append({"path": relative_text, "sha256": digest, "size": len(data), "owner": "ggml-hrx"})
        source_text = data.decode("utf-8")
        exports.extend(parse_exports(source_text, relative_text))
        merge_amdgpu_targets(target_selectors, parse_amdgpu_targets(source_text))

    resolve_export_variants(exports, target_selectors)

    for item in exports:
        recipe = compile_recipe(str(item["source"]))
        item["compile_recipe"] = recipe
        item["compile_dependencies"] = list(recipe["library_sources"])

    required_modules: set[str] = set()

    def require_module(name: str) -> None:
        if name in required_modules:
            return
        if name not in modules_by_name:
            raise RuntimeError(f"selected kernel recipe references unknown link module {name}")
        required_modules.add(name)
        for library in modules_by_name[name]["libraries"]:
            if str(library).startswith(":"):
                require_module(str(library)[1:])

    required_files = set(CORPUS_FILES)
    for item in exports:
        if str(item["source"]).startswith("../"):
            continue
        recipe = item["compile_recipe"]
        link_module = str(recipe.get("link_module", ""))
        if link_module:
            require_module(link_module)
        required_files.update(str(path) for path in recipe["primary_sources"])
        required_files.update(str(path) for path in recipe["library_sources"])

    mirrored_files = set(CORPUS_FILES) | set(OWNED_CORPUS_FILES)
    for relative_text in sorted(required_files - mirrored_files):
        relative = pathlib.Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"kernel recipe escapes the source corpus: {relative_text}")
        source = source_directory / relative
        if not source.is_file():
            raise RuntimeError(f"missing required kernel dependency: {source}")
        data = read_upstream_source(source, relative_text)
        digest = sha256(data)
        upstream_aggregate.update(relative_text.encode())
        upstream_aggregate.update(b"\0")
        upstream_aggregate.update(bytes.fromhex(digest))
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        file_rows.append({"path": relative_text, "sha256": digest, "size": len(data)})

    link_modules = [
        module for module in all_link_modules
        if str(module["name"]) in required_modules
    ]
    plan_cases = [
        case for case in all_plan_cases
        if (str(case.get("link_module", "")) in required_modules or
            str(case.get("source", "")) in required_files)
    ]
    plan_cases.extend([
        {
            "name": "owned_token_embedding_decode_plan_test",
            "args": ["$(location ../qwen_owned/token_embedding_q4k.loom)",
                     "--benchmark=@qwen_token_embedding_q4k_decode", "--dry-run",
                     "--output-format=jsonl"],
            "source": "../qwen_owned/token_embedding_q4k.loom",
        },
        {
            "name": "owned_token_embedding_prefill_plan_test",
            "args": ["$(location ../qwen_owned/token_embedding_q4k.loom)",
                     "--benchmark=@qwen_token_embedding_q4k_prefill_512", "--dry-run",
                     "--output-format=jsonl"],
            "source": "../qwen_owned/token_embedding_q4k.loom",
        },
        {
            "name": "owned_attention_context_base_capture_plan_test",
            "args": ["$(location ../qwen_owned/attention_state_initialize.loom)",
                     "--benchmark=@qwen_attention_context_base_capture_benchmark", "--dry-run",
                     "--output-format=jsonl"],
            "source": "../qwen_owned/attention_state_initialize.loom",
            "owner": "ggml-hrx",
        },
        {
            "name": "owned_attention_decode_state_initialize_plan_test",
            "args": ["$(location ../qwen_owned/attention_state_initialize.loom)",
                     "--benchmark=@qwen_attention_decode_state_initialize_benchmark", "--dry-run",
                     "--output-format=jsonl"],
            "source": "../qwen_owned/attention_state_initialize.loom",
            "owner": "ggml-hrx",
        },
        {
            "name": "owned_attention_metadata_prefill_plan_test",
            "args": ["$(location ../qwen_owned/attention_metadata.loom)",
                     "--benchmark=@qwen_attention_metadata_prefill_512", "--dry-run",
                     "--output-format=jsonl"],
            "source": "../qwen_owned/attention_metadata.loom",
        },
        {
            "name": "owned_gather_add_plan_test",
            "args": ["$(location ../hrx_owned/gather_add_f32.loom)",
                     "--benchmark=@ggml_gather_add_noncontiguous", "--dry-run",
                     "--output-format=jsonl"],
            "source": "../hrx_owned/gather_add_f32.loom",
            "owner": "ggml-hrx",
        },
        {
            "name": "owned_scale_bias_plan_test",
            "args": ["$(location ../hrx_owned/add_f32.loom)",
                     "--benchmark=@ggml_scale_bias_f32_small",
                     "--config=ggml.scale.scale=2.0", "--config=ggml.scale.bias=1.0",
                     "--dry-run", "--output-format=jsonl"],
            "source": "../hrx_owned/add_f32.loom",
            "owner": "ggml-hrx",
        },
    ])

    upstream_digest = upstream_aggregate.hexdigest()
    owned_digest = owned_aggregate.hexdigest()
    combined_aggregate = hashlib.sha256()
    combined_aggregate.update(bytes.fromhex(upstream_digest))
    combined_aggregate.update(bytes.fromhex(owned_digest))

    manifest = {
        "schema": "ggml-hrx-qwen-kernel-corpus-v2",
        "upstream_repository": upstream_repository(source_root),
        "upstream_revision": revision,
        "source_subdirectory": SOURCE_SUBDIR.as_posix(),
        "qwen_endpoint_source_subdirectory": QWEN_ENDPOINT_SOURCE_SUBDIR.as_posix(),
        "corpus_sha256": combined_aggregate.hexdigest(),
        "upstream_corpus_sha256": upstream_digest,
        "owned_corpus_sha256": owned_digest,
        "build_bazel_sha256": sha256(build_data),
        "files": file_rows,
        "exports": sorted(exports, key=lambda item: (str(item["symbol"]), str(item["source"]))),
        "link_modules": link_modules,
        "plan_cases": plan_cases,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def trees_equal(lhs: pathlib.Path, rhs: pathlib.Path) -> bool:
    lhs_files = sorted(path.relative_to(lhs) for path in lhs.rglob("*") if path.is_file())
    rhs_files = sorted(path.relative_to(rhs) for path in rhs.rglob("*") if path.is_file())
    return lhs_files == rhs_files and all((lhs / path).read_bytes() == (rhs / path).read_bytes() for path in lhs_files)


def external_files() -> tuple[pathlib.Path, ...]:
    return tuple(pathlib.Path(name) for _, name in QWEN_ENDPOINT_FILES) + tuple(
        pathlib.Path(name) for name in OWNED_FILES)


def external_files_equal(generated_qwen_moe: pathlib.Path, destination_qwen_moe: pathlib.Path) -> bool:
    generated_kernel_root = generated_qwen_moe.parent
    destination_kernel_root = destination_qwen_moe.parent
    for relative in external_files():
        generated = generated_kernel_root / relative
        destination = destination_kernel_root / relative
        if not destination.is_file() or generated.read_bytes() != destination.read_bytes():
            return False
    return True


def copy_external_files(generated_qwen_moe: pathlib.Path, destination_qwen_moe: pathlib.Path) -> None:
    generated_kernel_root = generated_qwen_moe.parent
    destination_kernel_root = destination_qwen_moe.parent
    for relative in external_files():
        source = generated_kernel_root / relative
        target = destination_kernel_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hrx-source", type=pathlib.Path, required=True)
    parser.add_argument("--destination", type=pathlib.Path, required=True)
    parser.add_argument("--expect-revision")
    parser.add_argument("--loom-format", type=pathlib.Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="hrx-qwen-corpus-") as temporary:
        generated = pathlib.Path(temporary) / args.destination.name
        construct(
            args.hrx_source.resolve(), generated, args.expect_revision,
            args.loom_format.resolve())
        if args.check:
            if (not args.destination.is_dir() or not trees_equal(generated, args.destination) or
                    not external_files_equal(generated, args.destination)):
                print("mirrored Qwen kernel corpus is stale", file=sys.stderr)
                return 1
            return 0
        args.destination.mkdir(parents=True, exist_ok=True)
        for child in args.destination.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        shutil.copytree(generated, args.destination, dirs_exist_ok=True)
        copy_external_files(generated, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
