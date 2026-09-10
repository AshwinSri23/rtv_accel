# Standard out-of-tree kbuild Makefile.
#
# `make` here only COMPILES the module against the host's kernel headers --
# it never loads anything, so it is always safe to run directly on the host.
# Loading (insmod) must only ever happen inside a virtme-ng VM; see
# driver/README.md.

obj-m += rtv_accel.o

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f test_rtv_accel

# Plain userspace program (not a kernel object) -- exercises the ioctl
# protocol from driver/rtv_accel_ioctl.h.  Building it doesn't touch the
# kernel at all; only running it inside the VM does.
test_rtv_accel: test_rtv_accel.c rtv_accel_ioctl.h
	$(CC) -Wall -Wextra -O2 -o $@ test_rtv_accel.c

.PHONY: all clean
