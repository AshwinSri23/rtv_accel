// ─── test_rtv_accel: userspace exerciser for the ioctl handshake ────────────
//
// A shell can't issue ioctl() calls, so this is a real program.  It forks:
// the child opens the device and blocks in RTV_IOC_WAIT_FOR_JOB *before* any
// job exists, proving the wait queue genuinely parks the calling process
// rather than busy-polling; the parent sleeps briefly (so the child is
// certainly already blocked), then issues RTV_IOC_START, which should wake
// the child via wake_up_interruptible() in the kernel.  The child then
// issues RTV_IOC_JOB_DONE, which should wake the parent's own blocking
// RTV_IOC_WAIT_DONE call the same way.
//
// This is the actual mechanism described for how a client and a "device-side"
// daemon would hand a job back and forth once a real daemon exists -- this
// test plays both roles from one program to prove the kernel-side plumbing
// before anything real depends on it.
//
// ────────────────────────────────────────────────────────────────────────────

#include "rtv_accel_ioctl.h"
#include <fcntl.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <unistd.h>

#define DEVICE_PATH "/dev/rtv_accel"
#define TEST_JOB_ID 42

// Plays the daemon role: blocks for a job, "does the work" (there is none
// yet), and reports completion.
static int run_daemon_child(void)
{
    int fd = open(DEVICE_PATH, O_RDWR);
    if (fd < 0) { perror("child: open"); return 1; }

    struct rtv_job job;
    printf("child: blocking on WAIT_FOR_JOB...\n");
    if (ioctl(fd, RTV_IOC_WAIT_FOR_JOB, &job) < 0) {
        perror("child: WAIT_FOR_JOB");
        close(fd);
        return 1;
    }
    printf("child: woke up, job_id=%u\n", job.job_id);

    struct rtv_result result;
    result.job_id = job.job_id;
    result.status = 0;
    if (ioctl(fd, RTV_IOC_JOB_DONE, &result) < 0) {
        perror("child: JOB_DONE");
        close(fd);
        return 1;
    }
    printf("child: signaled JOB_DONE\n");

    close(fd);
    return (job.job_id == TEST_JOB_ID) ? 0 : 1;
}

// Plays the client role: submits a job, blocks for its result.
static int run_client_parent(pid_t child_pid)
{
    sleep(1);   // give the child a real chance to be parked in WAIT_FOR_JOB

    int fd = open(DEVICE_PATH, O_RDWR);
    if (fd < 0) { perror("parent: open"); return 1; }

    struct rtv_job job;
    job.job_id = TEST_JOB_ID;
    printf("parent: issuing START job_id=%u\n", job.job_id);
    if (ioctl(fd, RTV_IOC_START, &job) < 0) {
        perror("parent: START");
        close(fd);
        return 1;
    }

    struct rtv_result result;
    printf("parent: blocking on WAIT_DONE...\n");
    if (ioctl(fd, RTV_IOC_WAIT_DONE, &result) < 0) {
        perror("parent: WAIT_DONE");
        close(fd);
        return 1;
    }
    printf("parent: woke up, job_id=%u status=%u\n", result.job_id, result.status);
    close(fd);

    int child_status = 0;
    waitpid(child_pid, &child_status, 0);
    const int child_ok = WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0;

    const int ok = child_ok && result.job_id == TEST_JOB_ID && result.status == 0;
    printf("\n%s\n", ok ? "PASS -- ioctl wait-queue round trip correct"
                        : "FAIL -- mismatch or child error");
    return ok ? 0 : 1;
}

int main(void)
{
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }

    if (pid == 0) {
        const int rc = run_daemon_child();
        // _exit() (not exit()) to skip atexit handlers inherited from the
        // parent, but that also skips stdio flushing -- and under vng's
        // piped (non-TTY) output, stdout is fully buffered rather than
        // line-buffered, so the child's printf output would otherwise be
        // silently dropped.  Flush explicitly before the unflushed _exit().
        fflush(stdout);
        _exit(rc);
    }

    return run_client_parent(pid);
}
