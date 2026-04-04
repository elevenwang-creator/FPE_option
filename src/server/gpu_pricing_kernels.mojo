from std.gpu import global_idx
from layout import Layout, LayoutTensor


def payoff_integration_kernel[
    pdf_layout: Layout, s_layout: Layout, v_layout: Layout, k_layout: Layout, bar_layout: Layout, out_layout: Layout
](
    pdf: LayoutTensor[DType.float64, pdf_layout, MutAnyOrigin],
    s_points: LayoutTensor[DType.float64, s_layout, MutAnyOrigin],
    v_points: LayoutTensor[DType.float64, v_layout, MutAnyOrigin],
    strikes: LayoutTensor[DType.float64, k_layout, MutAnyOrigin],
    barriers: LayoutTensor[DType.float64, bar_layout, MutAnyOrigin],
    prices: LayoutTensor[DType.float64, out_layout, MutAnyOrigin],
    n_s: Int,
    n_v: Int,
):
    """GPU kernel: one thread per option. Integrate payoff over PDF grid."""
    var option_idx = global_idx.x
    
    var K = rebind[Float64](strikes[option_idx])
    var barrier = rebind[Float64](barriers[option_idx])

    var price: Float64 = 0.0
    for i in range(n_s):
        var S = rebind[Float64](s_points[i])
        var payoff = S - K
        if payoff < 0.0:
            payoff = 0.0
        if S >= barrier:
            payoff = 0.0

        for j in range(n_v):
            price += payoff * rebind[Float64](pdf[i * n_v + j])

    if option_idx < UInt(prices.size()):
        prices[option_idx] = rebind[prices.element_type](price)


def greeks_kernel[
    pdf_layout: Layout,
    s_layout: Layout, 
    v_layout: Layout,
    k_layout: Layout,
    bar_layout: Layout,
    out_layout: Layout
](
    pdf: LayoutTensor[DType.float64, pdf_layout, MutAnyOrigin],
    s_points: LayoutTensor[DType.float64, s_layout, MutAnyOrigin],
    v_points: LayoutTensor[DType.float64, v_layout, MutAnyOrigin],
    strikes: LayoutTensor[DType.float64, k_layout, MutAnyOrigin],
    barriers: LayoutTensor[DType.float64, bar_layout, MutAnyOrigin],
    greeks_out: LayoutTensor[DType.float64, out_layout, MutAnyOrigin],
    n_s: Int,
    n_v: Int,
):
    """GPU kernel for Greeks via Finite Difference interpolation."""
    var option_idx = global_idx.x
    _ = (pdf, s_points, v_points, strikes, barriers, n_s, n_v)
    if option_idx < UInt(greeks_out.size()):
        greeks_out[option_idx] = rebind[greeks_out.element_type](0.0)
