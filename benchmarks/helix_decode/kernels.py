"""Fixed FP32 inference-only routed FFN example; not a library API."""
import mlx.core as mx

D, F, E, K = 1024, 2048, 4, 2


def pack(projections):
    """Stack frozen per-expert [out,in] weights. Rebuild after any weight change."""
    shapes = ((F, D), (F, D), (D, F))
    if len(projections) != 3 or any(
        len(group) != E or any(w.shape != shape or w.dtype != mx.float32 for w in group)
        for group, shape in zip(projections, shapes)
    ):
        raise ValueError("expected four FP32 gate, up and down matrices")
    return tuple(mx.contiguous(mx.stack(group)) for group in projections)


def top_two(logits):
    """Synthetic router: returns in-range GPU IDs and mixture weights, no host read."""
    if logits.shape != (E,) or logits.dtype != mx.float32:
        raise ValueError("expected four FP32 routing logits")
    ix = mx.contiguous(mx.argsort(logits)[-K:])
    return ix, mx.contiguous(mx.softmax(logits[ix]))


def selected_decode(x, packed, logits):
    """Single token only. Obtain packs from pack(); routing here is illustrative."""
    if x.shape != (D,) or x.dtype != mx.float32:
        raise ValueError("expected one 1024-element FP32 vector")
    if len(packed) != 3 or any(w.shape != s or w.dtype != mx.float32 for w, s in
                              zip(packed, ((E,F,D), (E,F,D), (E,D,F)))):
        raise ValueError("bad frozen pack")
    ix, alpha = top_two(logits)
    return _dispatch(mx.contiguous(x), packed, ix, alpha)


def _dispatch(x, packed, ix, alpha):
    # Internal: ix comes from top_two(), so ids are in [0,3]. No CPU selection.
    wg, wu, wd = packed
    h = GATE(inputs=[x, wg, wu, ix], grid=(32*F, K, 1), threadgroup=(128,1,1),
             output_shapes=[(K,F)], output_dtypes=[mx.float32])[0]
    return DOWN(inputs=[h, wd, ix, alpha], grid=(32*D, 1, 1), threadgroup=(128,1,1),
                output_shapes=[(D,)], output_dtypes=[mx.float32])[0]


GATE=mx.fast.metal_kernel(name="helix_decode_two_swiglu",input_names=["x","g","u","ix"],output_names=["h"],source="""
    uint lane=thread_position_in_grid.x % 32, row=thread_position_in_grid.x/32, slot=thread_position_in_grid.y;
    uint off=(uint(ix[slot])*2048+row)*1024;
    auto xv=(device const float4*)x;
    auto gv=(device const float4*)(g+off); auto uv=(device const float4*)(u+off);
    float a=0,b=0;
    for(uint j=lane;j<256;j+=32){ float4 v=xv[j]; a+=dot(v,gv[j]); b+=dot(v,uv[j]); }
    a=simd_sum(a); b=simd_sum(b);
    if(lane==0)h[slot*2048+row]=(a/(1.0f+exp(-a)))*b;
""")
DOWN=mx.fast.metal_kernel(name="helix_decode_two_down_mix",input_names=["h","wt","ix","alpha"],output_names=["y"],source="""
    uint lane=thread_position_in_grid.x%32, row=thread_position_in_grid.x/32;
    float sum=0;
    for(uint slot=0;slot<2;slot++){
        auto v=(device const float4*)(h+slot*2048);
        auto p=(device const float4*)(wt+(uint(ix[slot])*1024+row)*2048);
        float s=0; for(uint j=lane;j<512;j+=32)s+=dot(v[j],p[j]);
        sum+=simd_sum(s)*alpha[slot];
    }
    if(lane==0)y[row]=sum;
""")
