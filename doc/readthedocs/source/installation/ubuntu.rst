.. _sec-installation-ubuntu:

.. image:: ../_static/installation/ubuntu-logo.svg
   :align: right
   :width: 100 px

Install on Ubuntu
=================

**Supported versions**: 14.04, 15.04, 15.10, 16.04

System setup
------------

.. code-block:: bash

   sudo add-apt-repository ppa:bqlabs/horus-dev
   sudo apt-get update

Official versions are hosted in **ppa:bqlabs/horus**: `PPA Horus`_.
Alpha, beta and rc versions are hosted in **ppa:bqlabs/horus-dev**: `PPA Horus dev`_.

.. note::

   A `custom OpenCV`_ version is used, because of the next `reasons`_.



Install Horus
-------------

This command installs all the dependencies, including custom OpenCV libraries.

.. code-block:: bash

   sudo apt-get install horus

.. note::

    If user has no access to serial port, execute ``sudo usermod -a -G dialout $USER`` and reboot.


Update Horus
------------

If there is a new release just execute

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install horus

.. _PPA Horus: https://launchpad.net/~bqlabs/+archive/ubuntu/horus/
.. _PPA Horus dev: https://launchpad.net/~bqlabs/+archive/ubuntu/horus-dev/
.. _custom OpenCV: https://github.com/LibreScanner/opencv
.. _reasons: https://github.com/LibreScanner/opencv/wiki
