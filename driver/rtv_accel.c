// ─── rtv_accel: RTV-Accel NPU character device driver ────────────────────────
//
// Part 3: adds the ioctl control-plane handshake (rtv_accel_ioctl.h) --
// START/WAIT_FOR_JOB/JOB_DONE/WAIT_DONE, backed by two wait queues.  This is
// the mechanism a client and a userspace "device-side" daemon would use to
// hand a job back and forth through the kernel: the kernel never calls into
// either process directly, it only ever unblocks a syscall one of them is
// already parked in (wait_event_interruptible / wake_up_interruptible).
//
// State is module-global, not per-fd private data: a client and a daemon
// opening the device in two separate processes must see the same job/result
// state, since they're coordinating through what simulates one physical
// device with one set of "registers" -- not two independent buffers.
//
// No DMA buffer yet (that's the next part) -- job_id/status are placeholders
// that only prove the wait-queue mechanism and the data path across the
// ioctl boundary, not real work.
//
// Tested exclusively inside a virtme-ng VM snapshot; see driver/README.md.
//
// ────────────────────────────────────────────────────────────────────────────

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/err.h>
#include <linux/mutex.h>
#include <linux/wait.h>
#include <linux/uaccess.h>
#include "rtv_accel_ioctl.h"

#define DEVICE_NAME "rtv_accel"
#define CLASS_NAME  "rtv_accel"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("rtv_accel");
MODULE_DESCRIPTION("RTV-Accel NPU character device driver");
MODULE_VERSION("0.3");

static dev_t          rtv_devt;
static struct cdev    rtv_cdev;
static struct class  *rtv_class;
static struct device *rtv_device;

// ── Job/result state, shared across every fd this device is opened from ──────

static DEFINE_MUTEX(rtv_lock);
static DECLARE_WAIT_QUEUE_HEAD(rtv_job_wq);
static DECLARE_WAIT_QUEUE_HEAD(rtv_done_wq);

static bool  job_pending;
static u32   pending_job_id;

static bool  job_done;
static u32   done_job_id;
static u32   done_status;

static int rtv_open(struct inode *inode, struct file *file)
{
    pr_info("rtv_accel: opened\n");
    return 0;
}

static int rtv_release(struct inode *inode, struct file *file)
{
    pr_info("rtv_accel: closed\n");
    return 0;
}

static long rtv_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    void __user *argp = (void __user *)arg;

    switch (cmd) {
    case RTV_IOC_START: {
        struct rtv_job req;
        if (copy_from_user(&req, argp, sizeof(req)))
            return -EFAULT;

        mutex_lock(&rtv_lock);
        if (job_pending) {
            mutex_unlock(&rtv_lock);
            return -EBUSY;   // one job in flight at a time, for now
        }
        pending_job_id = req.job_id;
        job_pending = true;
        mutex_unlock(&rtv_lock);

        wake_up_interruptible(&rtv_job_wq);
        pr_info("rtv_accel: START job_id=%u\n", req.job_id);
        return 0;
    }

    case RTV_IOC_WAIT_FOR_JOB: {
        struct rtv_job resp;
        int ret = wait_event_interruptible(rtv_job_wq, job_pending);
        if (ret)
            return ret;   // -ERESTARTSYS if interrupted by a signal

        mutex_lock(&rtv_lock);
        resp.job_id = pending_job_id;
        job_pending = false;   // consumed -- next START may proceed
        mutex_unlock(&rtv_lock);

        if (copy_to_user(argp, &resp, sizeof(resp)))
            return -EFAULT;
        pr_info("rtv_accel: WAIT_FOR_JOB returned job_id=%u\n", resp.job_id);
        return 0;
    }

    case RTV_IOC_JOB_DONE: {
        struct rtv_result req;
        if (copy_from_user(&req, argp, sizeof(req)))
            return -EFAULT;

        mutex_lock(&rtv_lock);
        done_job_id = req.job_id;
        done_status = req.status;
        job_done = true;
        mutex_unlock(&rtv_lock);

        wake_up_interruptible(&rtv_done_wq);
        pr_info("rtv_accel: JOB_DONE job_id=%u status=%u\n", req.job_id, req.status);
        return 0;
    }

    case RTV_IOC_WAIT_DONE: {
        struct rtv_result resp;
        int ret = wait_event_interruptible(rtv_done_wq, job_done);
        if (ret)
            return ret;

        mutex_lock(&rtv_lock);
        resp.job_id = done_job_id;
        resp.status = done_status;
        job_done = false;   // consumed
        mutex_unlock(&rtv_lock);

        if (copy_to_user(argp, &resp, sizeof(resp)))
            return -EFAULT;
        pr_info("rtv_accel: WAIT_DONE returned job_id=%u status=%u\n",
                resp.job_id, resp.status);
        return 0;
    }

    default:
        return -ENOTTY;
    }
}

static const struct file_operations rtv_fops = {
    .owner          = THIS_MODULE,
    .open           = rtv_open,
    .release        = rtv_release,
    .unlocked_ioctl = rtv_ioctl,
};

static int __init rtv_accel_init(void)
{
    int ret;

    ret = alloc_chrdev_region(&rtv_devt, 0, 1, DEVICE_NAME);
    if (ret < 0) {
        pr_err("rtv_accel: alloc_chrdev_region failed: %d\n", ret);
        return ret;
    }

    cdev_init(&rtv_cdev, &rtv_fops);
    rtv_cdev.owner = THIS_MODULE;
    ret = cdev_add(&rtv_cdev, rtv_devt, 1);
    if (ret < 0) {
        pr_err("rtv_accel: cdev_add failed: %d\n", ret);
        goto err_unregister_region;
    }

    rtv_class = class_create(CLASS_NAME);
    if (IS_ERR(rtv_class)) {
        ret = PTR_ERR(rtv_class);
        pr_err("rtv_accel: class_create failed: %d\n", ret);
        goto err_del_cdev;
    }

    rtv_device = device_create(rtv_class, NULL, rtv_devt, NULL, DEVICE_NAME);
    if (IS_ERR(rtv_device)) {
        ret = PTR_ERR(rtv_device);
        pr_err("rtv_accel: device_create failed: %d\n", ret);
        goto err_destroy_class;
    }

    pr_info("rtv_accel: module loaded, /dev/%s ready (major=%d minor=%d)\n",
            DEVICE_NAME, MAJOR(rtv_devt), MINOR(rtv_devt));
    return 0;

err_destroy_class:
    class_destroy(rtv_class);
err_del_cdev:
    cdev_del(&rtv_cdev);
err_unregister_region:
    unregister_chrdev_region(rtv_devt, 1);
    return ret;
}

static void __exit rtv_accel_exit(void)
{
    device_destroy(rtv_class, rtv_devt);
    class_destroy(rtv_class);
    cdev_del(&rtv_cdev);
    unregister_chrdev_region(rtv_devt, 1);
    pr_info("rtv_accel: module unloaded\n");
}

module_init(rtv_accel_init);
module_exit(rtv_accel_exit);
