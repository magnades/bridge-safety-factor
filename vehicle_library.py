# vehicle_library.py
PREDEFINED_VEHICLES = {
    "Portugal": {
        "n_axles": 3,
        "axle_loads": [200, 200, 200],              # kN
        "axle_positions": [1.5 , 1.5]               # spacing between axles
    },
    "Brazil ": {
                "n_axles": 3,
                "axle_loads": [75, 75, 75],
                "axle_positions": [1.5, 1.5]
            },

    "Tandem": {
        "n_axles": 2,
        "axle_loads": [100, 100],
        "axle_positions": [1.0]                     # one spacing
    },
    "Truck A": {
        "n_axles": 4,
        "axle_loads": [80, 120, 120, 80],           # kN
        "axle_positions": [2.5, 3., 3.]             # spacings between axles
    },
    "Truck B": {
        "n_axles": 4,
        "axle_loads": [100, 150, 150, 100],
        "axle_positions": [1.5, 1.5, 1.5]   # corrected list format
    },

}
