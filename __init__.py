"""
ComfyUI custom node: registers dpmpp_Nm_sde_gpu / dpmpp_Nm_heun_gpu (N=1..8)
as selectable entries in KSampler's sampler_name dropdown.

Install: drop this folder into ComfyUI/custom_nodes/, restart ComfyUI.
Then in KSampler (Advanced), pick e.g. "dpmpp_6m_sde_gpu" or
"dpmpp_3m_heun_gpu" from the sampler_name list.

See sampling_ext.py's module docstring for which orders are exact
reference reproductions (1-3) vs. experimental generalization (4-8), and
test_order_convergence.py for an empirical order-of-accuracy check.
"""

import comfy.samplers
from .sampling_ext import SAMPLER_FUNCS

_EXTRA_SAMPLERS = sorted(SAMPLER_FUNCS.keys())

# Register with ComfyUI's global sampler registry so they show up in the
# ordinary KSampler dropdown, not just in a dedicated custom node.
for _name, _func in SAMPLER_FUNCS.items():
    if _name not in comfy.samplers.SAMPLER_NAMES:
        comfy.samplers.SAMPLER_NAMES.append(_name)
    comfy.samplers.KSAMPLER_NAMES_TO_FUNCS = getattr(
        comfy.samplers, "KSAMPLER_NAMES_TO_FUNCS", {})
    # Monkeypatch the lookup ComfyUI uses internally to resolve a sampler
    # name string -> callable. ComfyUI's own `sampler_object()` dispatches
    # on `k_diffusion_sampling.__dict__[f"sample_{name}"]`, so we inject
    # into that module's namespace under the expected name instead of
    # relying on private registry internals that vary across versions.
    import comfy.k_diffusion.sampling as _kds
    setattr(_kds, f"sample_{_name}", _func)


class EmptyPassthroughNode:
    """
    No-op node. The real work above happens at import time (module
    registration). This node exists only so the extension shows up in
    ComfyUI's node list as confirmation it loaded; it has no required
    inputs/outputs you need to wire up.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "sampling/custom_sampling/dpmpp_extended"
    OUTPUT_NODE = True

    def noop(self):
        print(f"[dpmpp_extended] registered samplers: {_EXTRA_SAMPLERS}")
        return ()


NODE_CLASS_MAPPINGS = {
    "DPMPPExtendedStatus": EmptyPassthroughNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DPMPPExtendedStatus": "DPM++ Extended Samplers (status)",
}
