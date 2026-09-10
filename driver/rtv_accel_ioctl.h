// ─── rtv_accel ioctl protocol ────────────────────────────────────────────────
//
// Shared between the kernel module (rtv_accel.c) and any userspace program
// that talks to /dev/rtv_accel (test_rtv_accel.c today; a real client/daemon
// pair later). This is the control-plane handshake described in the driver
// design discussion: a client issues START with a job descriptor, a daemon
// blocked in WAIT_FOR_JOB wakes up and receives it, the daemon issues
// JOB_DONE with a result, and a client blocked in WAIT_DONE wakes up and
// receives that.
//
// job_id/status are placeholders for now -- there is no DMA buffer yet
// (that's the next part), so there is nothing real to describe. Once mmap
// lands, struct rtv_job will carry the DMA buffer offset/length the job
// operates on instead of an opaque id.
//
// ────────────────────────────────────────────────────────────────────────────

#ifndef RTV_ACCEL_IOCTL_H
#define RTV_ACCEL_IOCTL_H

#include <linux/ioctl.h>
#include <linux/types.h>

struct rtv_job {
    __u32 job_id;
};

struct rtv_result {
    __u32 job_id;
    __u32 status;   // 0 = success, nonzero = error code
};

#define RTV_IOC_MAGIC 'r'

// Client -> kernel: a job is ready.  -EBUSY if one is already pending
// (single job in flight at a time, for now).
#define RTV_IOC_START         _IOW(RTV_IOC_MAGIC, 1, struct rtv_job)

// Daemon -> kernel: block until a job is pending, then consume it.
#define RTV_IOC_WAIT_FOR_JOB  _IOR(RTV_IOC_MAGIC, 2, struct rtv_job)

// Daemon -> kernel: the job finished; wake anyone blocked in WAIT_DONE.
#define RTV_IOC_JOB_DONE      _IOW(RTV_IOC_MAGIC, 3, struct rtv_result)

// Client -> kernel: block until the job's result is posted, then consume it.
#define RTV_IOC_WAIT_DONE     _IOR(RTV_IOC_MAGIC, 4, struct rtv_result)

#endif
