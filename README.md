# FE65P2

 [![Build Status](https://travis-ci.org/SiLab-Bonn/fe65_p2.svg?branch=master)](https://travis-ci.org/SiLab-Bonn/fe65_p2)
 
DAQ for FE65P2 prototype based on [Basil](https://github.com/SiLab-Bonn/basil) framwork. Many files and ideas are taken from [pyBar](https://github.com/SiLab-Bonn/pyBAR) project.

## Instalation
Use [conda](http://conda.pydata.org) for python. See [.travis.yml](https://github.com/SiLab-Bonn/fe65_p2/blob/master/.travis.yml) for detail. 
- for USB support see [pySiLibUSB](https://github.com/SiLab-Bonn/pySiLibUSB)

## Usage
```
python fe65p2/scans/digital_scan.py
```

## TODO
- hdf5 file storage for configuration and status (attibutes)
- configuration loading
- scans
