# models
    run models:
    -inception v4: $ cd ~/models && python test_image_classification_incepv4.py
    -Resnet,VGG: $ cd ~/models && python test_image_classification.py
    -Lenet: $ cd ~/models && && python test_recognize_digits.py

    need the below set to avoid the HCC STATUS_CHECK Error: HSA_STATUS_ERROR_OUT_OF_RESOURCES (0x1008) at file:mcwamp_hsa.cpp

    sudo su
    echo "vm.max_map_count = 250000" > /etc/sysctl.d/mmap.conf
    exit
    sudo reboot
