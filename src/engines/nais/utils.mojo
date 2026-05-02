from std.random import randn
from std.memory import alloc

from engines.nais.nais_net import NaisNet


def _generate_brownian_paths(M: Int, N: Int, D: Int) -> List[List[List[Float64]]]:
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

def _flatten_mat(mut p: List[Float64], W: List[List[Float64]]):
    """Append all elements of a 2D weight matrix to the flat vector."""
    for i in range(len(W)):
        for j in range(len(W[i])):
            p.append(W[i][j])


def _flatten_vec(mut p: List[Float64], b: List[Float64]):
    """Append all elements of a bias vector to the flat vector."""
    for j in range(len(b)):
        p.append(b[j])


def _flatten_net_params(net: NaisNet) -> List[Float64]:
    """Serialize all network weights into a flat vector for optimization."""
    var p: List[Float64] = []

    # Layer 1
    _flatten_mat(p, net.layer1)
    _flatten_vec(p, net.layer1_b)

    # Layers 2-4
    _flatten_mat(p, net.layer2.W)
    _flatten_vec(p, net.layer2.b)
    _flatten_mat(p, net.layer3.W)
    _flatten_vec(p, net.layer3.b)
    _flatten_mat(p, net.layer4.W)
    _flatten_vec(p, net.layer4.b)

    # Skip connections
    _flatten_mat(p, net.layer2_input)
    _flatten_vec(p, net.layer2_input_b)
    _flatten_mat(p, net.layer3_input)
    _flatten_vec(p, net.layer3_input_b)
    _flatten_mat(p, net.layer4_input)
    _flatten_vec(p, net.layer4_input_b)

    # Output layers
    _flatten_mat(p, net.layer5)
    _flatten_vec(p, net.layer5_b)
    _flatten_mat(p, net.layer6)
    _flatten_vec(p, net.layer6_b)

    return p^


def _unflatten_mat(p: List[Float64], idx: Int, mut W: List[List[Float64]]) -> Int:
    """Read elements from flat vector into a 2D weight matrix. Returns updated idx."""
    var pos = idx
    for i in range(len(W)):
        for j in range(len(W[i])):
            W[i][j] = p[pos]
            pos += 1
    return pos


def _unflatten_vec(p: List[Float64], idx: Int, mut b: List[Float64]) -> Int:
    """Read elements from flat vector into a bias vector. Returns updated idx."""
    var pos = idx
    for j in range(len(b)):
        b[j] = p[pos]
        pos += 1
    return pos
