# **ECOdiagnostics**

A python diagnostics package for the Energy Cycle of the Ocean. 
It builds on the descriptions of energy in the ocean as described in [Rosenthal2023]. It is designed to be applied to numerical ocean model simulation data generated with nemo, but can be modified at will. 

This package includes the following three major components:

### **ECOdiagnostics**
The main package which provides the functions to diagnose energy cycle variables. Further, a large set of functions which help analysing numerical data is provided.

### **ECOprocessing**
A subpackage which provides a processing script to prepare the original numerical output from nemo and process selected energy cycle variables.

### **ECOanalysis**
A subpackage which provides notebooks used to analyse and visualize data processed with ECOprocessing.

