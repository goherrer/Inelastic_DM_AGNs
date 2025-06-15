This folder contains exclusions where attenuation has been taken into account. The SK detector is treated as 1 km underground, and the Earth is treated as the PREM model.

Files with "Rescaled" in the name are constraints with attenuation accounted for.

Files with "Upper" are the couplings where attenuation becomes too strong. Each "Rescaled" file should have a corresponding "Upper" file (there may be multiple "Rescaled" per "Upper" file.) If the "Rescaled" coupling is greater than or equal to the "Upper" coupling, then that means no constraints can be set at that mass and mass splitting.

Some files have the mediator mass and mass splitting fixed to a ratio of the dark matter mass.  Other files fix the mediator mass and vary mass and mass splitting. The name should indicate which is which.

If the type of detector scattering is not specified, it is meant to be electron scattering.

I recommend viewing these files in Excel so that the spacing is done properly.
