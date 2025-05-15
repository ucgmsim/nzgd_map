"""
Start the nzgd_map Flask app.
"""

import nzgd_map  # module containing the Flask nzgd_map

app = nzgd_map.create_app()

if __name__ == "__main__":
    app.run()
