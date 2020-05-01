# Horus installation in Debian ![][debian-logo]

[return to Home](../../README.md)

### Not supported for version 0.2

Versions: 8

However it can be installed following the next instructions.

First you need to build and install our [custom OpenCV](https://github.com/LibreScanner/opencv/wiki/Build)

Then, generate the executable file following the next [instructions](../development/ubuntu.md).

```bash
bash package.sh debian -i
```

If user has no access to serial port, execute:

```bash
sudo usermod -a -G dialout $USER
sudo reboot
```

[debian-logo]: ../images/debian.png
