# **ECOdiagnostics**

ECOdiagnostics is the core component which builds on the xgcm-package to extend analysis of numerical General Circulation Models (GCM) focussing on the ocean energy cycle.

>  **Note**   
This is not a complete implementation of all aspects of the ocean energy cycle, but rather a set of implementations to provide the tools to produce the contents of [Rosenthal2023]

### Structure

- **operations.py:** An extended class in the sense of a xgcm-grid to include useful analysis tools.
- **diagnostics:** 
  - **properties.py:** Physical properties
  - **energetics.py:** Functions to compute energy reservoirs/trends
  - **power.py:** Wind power inputs as in [Roquet]
  - **transport.py** Transport functions as MOC / Ekman processes

Every diagnostic includes a class which inherits information of the grid provided, e.g. a property class uses the operations class and additional information of the coords

``` python
import ECOdiagnostics as eco
_coords = { 'X': ds.glamt,
            'Y': ds.gphit, 
            'Z': ds.depth_1d }
properties= eco.Properties(grid_ops, _coords)
```
