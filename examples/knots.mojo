from numerics.bspline.knots import GenerateKnots

def main():
    var knots = GenerateKnots(
            n=38,
            degree=3,
            method="non-uniform",
            center=0.1,
            boundary=(50.0, 150.0),
            mean=60.0,
            std=0.1,
        )
    s_knots =   knots.generate_knots()
    print(s_knots)
    print("Knots Length:", len(s_knots))
