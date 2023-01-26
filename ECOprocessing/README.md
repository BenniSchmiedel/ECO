# **ECOprocessing**


ECOprocessing is a subpackage which provides a processing script to handle everything from 
the preparation of the raw nemo_output to processed data (with variables relevant to the study of the centre of gravity)

The configuration of the processing is provided in the subdirectory "Config/". Configurations customized to the simulation at hand can be provided in a subfolder. 
A test simulation EXP00_test is provided, with the configuration in "Config/EXP00_test/".<br>
 - **base.yml** is the main configuration, where all parameters are specified.<br>
 - **sub_test.yml** is a sub-configuration to change single parameters. <br>

In this case, the appropriate execution of the script is:

    python ECO_processing.py -p Configs/EXP00_test -c sub_test

**-p:** The path to the configuration in use<br>
**-c:** The name of the sub-configuration

>  **Important**   
The script provided is optimized for an idealized box-model configuration.<br> Processing of other nemo configuration require adaptation of the function ECO_processing.prepare_dataset(), which handles metrics and boundary masking.
