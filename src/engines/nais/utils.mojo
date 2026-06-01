from std.random import randn
from std.memory import alloc

from engines.nais.nais_net import NaisNet


def _generate_brownian_paths(
    M: Int, N: Int, D: Int
) -> List[List[List[Float64]]]:
    var out: List[List[List[Float64]]] = []
    var total = M * (N + 1) * D
    var buf = alloc[Float64](total)
    randn(buf, total)

    var idx = 0
    for _ in range(M):
        var path: List[List[Float64]] = []
        for _ in range(N + 1):
            var step: List[Float64] = []
            for _ in range(D):
                step.append(buf[idx])
                idx += 1
            path.append(step^)
        out.append(path^)
    buf.free()
    return out^


def _flatten_net_params(net: NaisNet) -> List[Float64]:
    var p: List[Float64] = []

    for v in net.layer1_T_flat:
        p.append(v)
    for v in net.layer1_b:
        p.append(v)

    for v in net.layer2.W_T_flat:
        p.append(v)
    for v in net.layer2.b:
        p.append(v)
    for v in net.layer3.W_T_flat:
        p.append(v)
    for v in net.layer3.b:
        p.append(v)
    for v in net.layer4.W_T_flat:
        p.append(v)
    for v in net.layer4.b:
        p.append(v)

    for v in net.layer2_input_T_flat:
        p.append(v)
    for v in net.layer2_input_b:
        p.append(v)
    for v in net.layer3_input_T_flat:
        p.append(v)
    for v in net.layer3_input_b:
        p.append(v)
    for v in net.layer4_input_T_flat:
        p.append(v)
    for v in net.layer4_input_b:
        p.append(v)

    for v in net.layer5_T_flat:
        p.append(v)
    for v in net.layer5_b:
        p.append(v)
    for v in net.layer6_T_flat:
        p.append(v)
    for v in net.layer6_b:
        p.append(v)

    return p^
