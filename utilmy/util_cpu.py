# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0601,E1123,W0614,F0401,E1120,E1101,E0611,W0702
"""
Launch processors and monitor the CPU, memory usage.
Maintain same level of processors over time.




"""
import csv
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from collections import namedtuple
from datetime import datetime
from time import sleep, time

import arrow
# non-stdlib imports
import psutil

from aapackage import util_log

############# Root folder #####################################################
VERSION = 1


#############Variable #########################################################

######### Logging ##############################################################
LOG_FILE = "zlog/" + util_log.create_logfilename(__file__)
APP_ID = util_log.create_appid(__file__)

logger = util_log.logger_setup(__name__, log_file=None, formatter=util_log.FORMATTER_4)


def log(*argv):
    """function log
    Args:
        *argv:   
    Returns:
        
    """
    logger.info(",".join([str(x) for x in argv]))


# log("Ok, test_log")
################################################################################


###############################################################################
def os_getparent(dir0):
    """function os_getparent
    Args:
        dir0:   
    Returns:
        
    """
    return os.path.abspath(os.path.join(dir0, os.pardir))


# noinspection PyTypeChecker
def ps_process_monitor_child(pid, logfile=None, duration=None, interval=None):
    """function ps_process_monitor_child
    Args:
        pid:   
        logfile:   
        duration:   
        interval:   
    Returns:
        
    """
    # We import psutil here so that the module can be imported even if psutil
    # is not present (for example if accessing the version)
    log("Monitoring Started for Process Id: %s" % str(pid))
    pr = psutil.Process(pid)

    # Record start time
    start_time = time.time()

    f = None
    if logfile:
        f = open(logfile, "w")
        f.write(
            "# {0:12s} {1:12s} {2:12s} {3:12s} {4:12s}\n".format(
                "Timestamp".center(12),
                "Elapsed time".center(12),
                "CPU (%)".center(12),
                "Real (MB)".center(12),
                "Virtual (MB)".center(12),
            )
        )

    try:

        # Start main event loop
        while True:

            # Find current time
            current_time = time.time()

            try:
                pr_status = ps_get_process_status(pr)
            except psutil.NoSuchProcess:
                break
            # Check if process status indicates we should exit
            if pr_status in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD]:
                log("Process finished ({0:.2f} seconds)".format(current_time - start_time))
                break

            # Check if we have reached the maximum time
            if duration is not None and current_time - start_time > duration:
                break

            # Get current CPU and memory
            try:
                current_cpu = ps_get_cpu_percent(pr)
                # noinspection PyTypeChecker
                current_mem = ps_get_memory_percent(pr)
            except Exception:
                break
            current_mem_real = current_mem.rss / 1024.0 ** 2
            current_mem_virtual = current_mem.vms / 1024.0 ** 2

            # Get information for children
            for child in ps_all_children(pr):
                try:
                    current_cpu += ps_get_cpu_percent(child)
                    current_mem = ps_get_memory_percent(child)
                except Exception:
                    continue
                current_mem_real += current_mem.rss / 1024.0 ** 2
                current_mem_virtual += current_mem.vms / 1024.0 ** 2

            if logfile:
                timestamp = str(arrow.utcnow().to("Japan").format("YYYYMMDD_HHmmss,"))
                f.write(
                    "{0:12} {1:12.3f} {2:12.3f} {3:12.3f} {4:12.3f}\n".format(
                        timestamp,
                        current_time - start_time,
                        current_cpu,
                        current_mem_real,
                        current_mem_virtual,
                    )
                )
                f.flush()

            sleep(interval)

    except KeyboardInterrupt:  # pragma: no cover
        pass

    if logfile:
        f.close()


def ps_wait_process_completion(subprocess_list, waitsec=10):
    """function ps_wait_process_completion
    Args:
        subprocess_list:   
        waitsec:   
    Returns:
        
    """
    for pid in subprocess_list:
        while True:
            try:
                pr = psutil.Process(pid)
                try:
                    pr_status = pr.status()
                except TypeError:  # psutil < 2.0
                    pr_status = pr.status
                except psutil.NoSuchProcess:  # pragma: no cover
                    break
                # Check if process status indicates we should exit
                if pr_status in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD]:
                    break
            except:
                break
            time.sleep(waitsec)


def ps_wait_ressourcefree(cpu_max=90, mem_max=90, waitsec=15):
    """function ps_wait_ressourcefree
    Args:
        cpu_max:   
        mem_max:   
        waitsec:   
    Returns:
        
    """
    # wait if computer resources are scarce.
    cpu, mem = 100, 100
    while cpu > cpu_max or mem > mem_max:
        cpu, mem = ps_get_computer_resources_usage()
        time.sleep(waitsec)


def ps_get_cpu_percent(process):
    """function ps_get_cpu_percent
    Args:
        process:   
    Returns:
        
    """
    try:
        return process.cpu_percent()
    except AttributeError:
        return process.get_cpu_percent()


def ps_get_memory_percent(process):
    """function ps_get_memory_percent
    Args:
        process:   
    Returns:
        
    """
    try:
        return process.memory_info()
    except AttributeError:
        return process.get_memory_info()


def ps_all_children(pr):
    """function ps_all_children
    Args:
        pr:   
    Returns:
        
    """
    processes = []
    children = []
    try:
        children = pr.children()
    except AttributeError:
        children = pr.get_children()
    except Exception:  # pragma: no cover
        pass

    for child in children:
        processes.append(child)
        processes += ps_all_children(child)
    return processes


def ps_get_process_status(pr):
    """function ps_get_process_status
    Args:
        pr:   
    Returns:
        
    """
    try:
        pr_status = pr.status()
    except TypeError:  # psutil < 2.0
        pr_status = pr.status
    except psutil.NoSuchProcess:  # pragma: no cover
        return psutil.STATUS_DEAD
        # raise psutil.NoSuchProcess
    return pr_status


def ps_process_isdead(pid):
    """function ps_process_isdead
    Args:
        pid:   
    Returns:
        
    """
    flag = 0
    try:
        pr = psutil.Process(pid)
        pr_status = ps_get_process_status(pr)
        if pr_status in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD]:
            flag = 1
    except:
        flag = 1

    return flag


def ps_get_computer_resources_usage():
    """function ps_get_computer_resources_usage
    Args:
    Returns:
        
    """
    cpu_used_percent = psutil.cpu_percent()

    mem_info = dict(psutil.virtual_memory()._asdict())
    # mem_used_percent = 100 - mem_info['available'] / mem_info['total']
    mem_used_percent = mem_info["percent"]

    return cpu_used_percent, mem_used_percent


def ps_find_procs_by_name(name=r"((.*/)?tasks.*/t.*/main\.(py|sh))", ishow=1, isregex=1):
    """ Return a list of processes matching 'name'.
        Regex (./tasks./t./main.(py|sh)|tasks./t.*/main.(py|sh))
        Condensed Regex to:
        ((.*/)?tasks.*/t.*/main\.(py|sh)) - make the characters before 'tasks' optional group.
    """
    ls = []
    for p in psutil.process_iter(["pid", "name", "exe", "cmdline"]):
        cmdline = " ".join(p.info["cmdline"]) if p.info["cmdline"] else ""
        if isregex:
            flag = re.match(name, cmdline, re.I)
        else:
            flag = name and name.lower() in cmdline.lower()

        if flag:
            ls.append({"pid": p.info["pid"], "cmdline": cmdline})

            if ishow > 0:
                log("Monitor", p.pid, cmdline)
    return ls


def os_launch(commands):
    """function os_launch
    Args:
        commands:   
    Returns:
        
    """
    processes = []
    for cmd in commands:
        try:
            p = subprocess.Popen(cmd, shell=False)
            processes.append(p.pid)
            log("Launched: ", p.pid, " ".join(cmd))
            sleep(1)

        except Exception as e:
            log(e)
    return processes


def ps_terminate(processes):
    """function ps_terminate
    Args:
        processes:   
    Returns:
        
    """
    for p in processes:
        pidi = p.pid
        try:
            os.kill(p.pid, 9)
            log("killed ", pidi)
        except Exception as e:
            log(e)
            try:
                os.kill(pidi, 9)
                log("killed ", pidi)
            except:
                pass


def os_extract_commands(csv_file, has_header=False):
    """function os_extract_commands
    Args:
        csv_file:   
        has_header:   
    Returns:
        
    """
    with open(csv_file, "r", newline="") as file:
        reader = csv.reader(file, skipinitialspace=True)
        if has_header:
            headers = next(reader)  # pass header
        commands = [row for row in reader]

    return commands


def ps_is_issue(p):
    """function ps_is_issue
    Args:
        p:   
    Returns:
        
    """
    global pars

    pdict = p.as_dict()
    pidi = p.pid

    Mb = 1024 ** 2
    log("Worker PID;CPU;RAM:", pidi, pdict["cpu_percent"], pdict["memory_full_info"][0] / Mb)

    try:
        if not psutil.pid_exists(pidi):
            log("Process has been killed ", pidi)
            return True

        elif pdict["status"] == "zombie":
            log("Process  zombie ", pidi)
            return True

        elif pdict["memory_full_info"][0] >= pars["max_memory"]:
            log("Process  max memory ", pidi)
            return True

        elif pdict["cpu_percent"] >= pars["max_cpu"]:
            log("Process MAX CPU ", pidi)
            return True

        else:
            return False
    except Exception as e:
        log(e)
        return True


def ps_net_send(tperiod=5):
    """function ps_net_send
    Args:
        tperiod:   
    Returns:
        
    """
    x0 = psutil.net_io_counters(pernic=False).bytes_sent
    t0 = time()
    sleep(tperiod)
    t1 = time()
    x1 = psutil.net_io_counters(pernic=False).bytes_sent
    return (x1 - x0) / (t1 - t0)


def ps_is_issue_system():
    """function ps_is_issue_system
    Args:
    Returns:
        
    """
    global net_avg, pars

    try:
        if psutil.cpu_percent(interval=5) > pars["cpu_usage_total"]:
            return True

        elif psutil.virtual_memory().available < pars["mem_available_total"]:
            return True

        else:
            return False

    except:
        return True


def monitor_maintain():
    """
       Launch processors and monitor the CPU, memory usage.
       Maintain same leve of processors over time.
    """
    global pars

    log("start monitoring", len(CMDS))
    cmds2 = []
    for cmd in CMDS:
        ss = shlex.split(cmd)
        cmds2.append(ss)

    processes = os_launch(cmds2)
    try:
        while True:
            has_issue = []
            ok_process = []
            log("N_process", len(processes))

            ### check global system  ##########################################
            if len(processes) == 0 or ps_is_issue_system():
                log("Reset all process")
                lpp = ps_find_procs_by_name(pars["proc_name"], 1)
                ps_terminate(lpp)
                processes = os_launch(cmds2)
                sleep(5)

            ## pid in process   ###############################################
            for pidi in processes:
                try:
                    p = psutil.Process(pidi)
                    log("Checking", p.pid)

                    if ps_is_issue(p):
                        has_issue.append(p)

                    else:
                        log("Process Fine ", pidi)
                        ok_process.append(p)

                except Exception as e:
                    log(e)

            ### Process with issues    ########################################
            for p in has_issue:
                try:
                    log("Relaunching", p.pid)
                    pcmdline = p.cmdline()
                    pidlist = os_launch([pcmdline])  # New process can start before

                    sleep(3)
                    ps_terminate([p])
                except:
                    pass

            ##### Check the number of  processes    ###########################
            sleep(5)
            lpp = ps_find_procs_by_name(pars["proc_name"], 1)

            log("Active process", len(lpp))
            if len(lpp) < pars["nproc"]:
                for i in range(0, pars["nproc"] - len(lpp)):
                    pidlist = os_launch([shlex.split(pars["proc_cmd"])])

            else:
                for i in range(0, len(lpp) - pars["nproc"]):
                    pidlist = ps_terminate([lpp[i]])

            sleep(5)
            lpp = ps_find_procs_by_name(pars["proc_name"], 0)
            processes = [x["pid"] for x in lpp]

            log("Waiting....")
            sleep(1)

    except Exception as e:
        log(e)


############ AZURE NODE #################################################################
#########################################################################################
"""
[  arg.logfolder, arg.name, arg.consumergroup, arg.input_topic,  
   arrow.utcnow().to('Japan').format("YYYYMMDD_HHmm_ss"),
     str(random.randrange(1000))])
# DIRCWD = os_getparent(os.path.dirname(os.path.abspath(__file__)))

"""
_DEFAULT_STATS_UPDATE_INTERVAL = 5
_OS_DISK = None
_USER_DISK = None
"""
_IS_PLATFORM_WINDOWS = platform.system() == 'Windows'
if _IS_PLATFORM_WINDOWS:
    _OS_DISK = 'C:/' # This is inverted on Cloud service
    _USER_DISK = 'D:/'
else:
    _OS_DISK = "/"
    _USER_DISK = '/mnt/resources'
    if not os.path.exists(_USER_DISK):
        _USER_DISK = '/mnt'
"""


def os_python_environment():  # pragma: no cover
    """function os_python_environment
    Args:
    Returns:
        
    """

def os_environment():
    """function os_environment
    Args:
    Returns:
        
    """
    return platform.platform()


def os_is_wndows():
    """function os_is_wndows
    Args:
    Returns:
        
    """
    return platform.system() == "Windows"


def np_avg(list):
    """function np_avg
    Args:
        list:   
    Returns:
        
    """
    return sum(list) / float(len(list))


def np_pretty_nb(num, suffix=""):
    """function np_pretty_nb
    Args:
        num:   
        suffix:   
    Returns:
        
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1000.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1000.0
    return "%.1f%s%s" % (num, "Yi", suffix)


NodeIOStats = namedtuple("NodeIOStats", ["read_bps", "write_bps"])


class NodeStats:
    """Persistent Task Stats class"""

    def __init__(
        self,
        num_connected_users=0,
        num_pids=0,
        cpu_count=0,
        cpu_percent=None,
        mem_total=0,
        mem_avail=0,
        swap_total=0,
        swap_avail=0,
        disk_io=None,
        disk_usage=None,
        net=None,
    ):
        """
        Map the attributes
        """
        self.num_connected_users = num_connected_users
        self.num_pids = num_pids
        self.cpu_count = cpu_count
        self.cpu_percent = cpu_percent
        self.mem_total = mem_total
        self.mem_avail = mem_avail
        self.swap_total = swap_total
        self.swap_avail = swap_avail
        self.disk_io = disk_io or NodeIOStats(0, 0)
        self.disk_usage = disk_usage or dict()
        self.net = net or NodeIOStats(0, 0)

    @property
    def mem_used(self):
        """
            Return the memory used
        """
        return self.mem_total - self.mem_avail


class IOThroughputAggregator:
    def __init__(self):
        """ IOThroughputAggregator:__init__
        Args:
        Returns:
           
        """
        self.last_timestamp = None
        self.last_read = 0
        self.last_write = 0

    def aggregate(self, cur_read, cur_write):
        """
            Aggregate with the new values
        """
        now = datetime.now()
        read_bps = 0
        write_bps = 0
        if self.last_timestamp:
            delta = (now - self.last_timestamp).total_seconds()
            read_bps = (cur_read - self.last_read) / delta
            write_bps = (cur_write - self.last_write) / delta

        self.last_timestamp = now
        self.last_read = cur_read
        self.last_write = cur_write

        return NodeIOStats(read_bps, write_bps)


class NodeStatsCollector:
    """
    Node Stats Manager class
    """

    def __init__(
        self,
        pool_id,
        node_id,
        refresh_interval=_DEFAULT_STATS_UPDATE_INTERVAL,
        app_insights_key=None,
    ):
        self.pool_id = pool_id
        self.node_id = node_id
        self.telemetry_client = None
        self.first_collect = True
        self.refresh_interval = refresh_interval

        self.disk = IOThroughputAggregator()
        self.network = IOThroughputAggregator()

        if (
            app_insights_key
            or "APP_INSIGHTS_INSTRUMENTATION_KEY" in os.environ
            or "APP_INSIGHTS_KEY" in os.environ
        ):
            key = (
                app_insights_key
                or os.environ.get("APP_INSIGHTS_INSTRUMENTATION_KEY")
                or os.environ.get("APP_INSIGHTS_KEY")
            )

            # log("Detected instrumentation key. Will upload stats to app insights")
            # self.telemetry_client = TelemetryClient(key)
            # context = self.telemetry_client.context
            # context.application.id = "AzureBatchInsights"
            # context.application.ver = VERSION
            # context.device.model = "BatchNode"
            # context.device.role_name = self.pool_id
            # context.device.role_instance = self.node_id
        else:
            log(
                "No instrumentation key detected. Cannot upload to app insights."
                + "Make sure you have the APP_INSIGHTS_INSTRUMENTATION_KEY environment variable setup"
            )

    def init(self):
        """
            Initialize the monitoring
        """
        # start cpu utilization monitoring, first value is ignored
        psutil.cpu_percent(interval=None, percpu=True)

    def _get_network_usage(self):
        """ NodeStatsCollector:_get_network_usage
        Args:
        Returns:
           
        """
        netio = psutil.net_io_counters()
        return self.network.aggregate(netio.bytes_recv, netio.bytes_sent)

    def _get_disk_io(self):
        """ NodeStatsCollector:_get_disk_io
        Args:
        Returns:
           
        """
        diskio = psutil.disk_io_counters()
        return self.disk.aggregate(diskio.read_bytes, diskio.write_bytes)

    def _get_disk_usage(self):
        """ NodeStatsCollector:_get_disk_usage
        Args:
        Returns:
           
        """
        disk_usage = dict()
        try:
            disk_usage[_OS_DISK] = psutil.disk_usage(_OS_DISK)
            disk_usage[_USER_DISK] = psutil.disk_usage(_USER_DISK)
        except Exception as e:
            logger.error("Could not retrieve user disk stats for {0}: {1}".format(_USER_DISK, e))
        return disk_usage

    def _sample_stats(self):
        """ NodeStatsCollector:_sample_stats
        Args:
        Returns:
           
        """
        # get system-wide counters
        mem = psutil.virtual_memory()
        disk_stats = self._get_disk_io()
        disk_usage = self._get_disk_usage()
        net_stats = self._get_network_usage()

        swap_total, _, swap_avail, _, _, _ = psutil.swap_memory()

        stats = NodeStats(
            cpu_count=psutil.cpu_count(),
            cpu_percent=psutil.cpu_percent(interval=None, percpu=True),
            num_pids=len(psutil.pids()),
            # Memory
            mem_total=mem.total,
            mem_avail=mem.available,
            swap_total=swap_total,
            swap_avail=swap_avail,
            # Disk IO
            disk_io=disk_stats,
            # Disk usage
            disk_usage=disk_usage,
            # Net transfer
            net=net_stats,
        )
        del mem
        return stats

    def _collect_stats(self):
        """
            Collect the stats and then send to app insights
        """
        # collect stats
        stats = self._sample_stats()

        if self.first_collect:
            self.first_collect = False
            return

        if stats is None:
            logger.error("Could not sample node stats")
            return

        if self.telemetry_client:
            self._send_stats(stats)
        else:
            self._log_stats(stats)

    def _send_stats(self, stats):
        """
            Retrieve the current stats and send to app insights
        """
        process = psutil.Process(os.getpid())

        logger.debug(
            "Uploading stats. Mem of this script: %d vs total: %d",
            process.memory_info().rss,
            stats.mem_avail,
        )
        client = self.telemetry_client

        for cpu_n in range(0, stats.cpu_count):
            client.track_metric("Cpu usage", stats.cpu_percent[cpu_n], properties={"Cpu #": cpu_n})

        for name, disk_usage in stats.disk_usage.items():
            client.track_metric("Disk usage", disk_usage.used, properties={"Disk": name})
            client.track_metric("Disk free", disk_usage.free, properties={"Disk": name})

        client.track_metric("Memory used", stats.mem_used)
        client.track_metric("Memory available", stats.mem_avail)
        client.track_metric("Disk read", stats.disk_io.read_bps)
        client.track_metric("Disk write", stats.disk_io.write_bps)
        client.track_metric("Network read", stats.net.read_bps)
        client.track_metric("Network write", stats.net.write_bps)
        self.telemetry_client.flush()

    def _log_stats(self, stats):
        """ NodeStatsCollector:_log_stats
        Args:
            stats:     
        Returns:
           
        """
        log("========================= Stats =========================")
        log("Cpu percent:            %d%% %s", np_avg(stats.cpu_percent), stats.cpu_percent)
        log(
            "Memory used:       %sB / %sB",
            np_pretty_nb(stats.mem_used),
            np_pretty_nb(stats.mem_total),
        )
        log(
            "Swap used:         %sB / %sB",
            np_pretty_nb(stats.swap_avail),
            np_pretty_nb(stats.swap_total),
        )
        log("Net read:               %sBs", np_pretty_nb(stats.net.read_bps))
        log("Net write:              %sBs", np_pretty_nb(stats.net.write_bps))
        log("Disk read:               %sBs", np_pretty_nb(stats.disk_io.read_bps))
        log("Disk write:              %sBs", np_pretty_nb(stats.disk_io.write_bps))
        log("Disk usage:")
        for name, disk_usage in stats.disk_usage.items():
            log("  - %s: %i/%i (%i%%)", name, disk_usage.used, disk_usage.total, disk_usage.percent)

        log("-------------------------------------")
        log("")

    def run(self):
        """
            Start collecting information of the system.
        """
        logger.debug("Start collecting stats for pool=%s node=%s", self.pool_id, self.node_id)
        while True:
            self._collect_stats()
            time.sleep(self.refresh_interval)


def monitor_nodes():
    """
    Main entry point for prism
    """
    # log basic info
    log("Python args: %s", sys.argv)
    log("Operating system: %s", os_environment())
    log("Cpu count: %s", psutil.cpu_count())

    pool_id = os.environ.get("AZ_BATCH_POOL_ID", "_test-pool-1")
    node_id = os.environ.get("AZ_BATCH_NODE_ID", "_test-node-1")

    # get and set event loop mode
    log("enabling event loop debug mode")

    app_insights_key = None
    if len(sys.argv) > 2:
        pool_id = sys.argv[1]
        node_id = sys.argv[2]
    if len(sys.argv) > 3:
        app_insights_key = sys.argv[3]

    # create node stats manager
    collector = NodeStatsCollector(pool_id, node_id, app_insights_key=app_insights_key)
    collector.init()
    collector.run()


def os_generate_cmdline():
    """function os_generate_cmdline
    Args:
    Returns:
        
    """
    Mb = 1024 ** 2
    pars = {
        "max_memory": 1500.0 * Mb,
        "max_cpu": 85.0,
        "proc_name": "streaming_couchbase_update_cli.py",
        "mem_available_total": 2000.0 * Mb,
        "cpu_usage_total": 98.0,
    }
    CMDS = [pars["proc_cmd"]] * pars["nproc"]
    return CMDS, pars


if __name__ == "__main__":
    ################## Initialization #########################################
    log(" Log check")

    CMDS, pars = os_generate_cmdline()

    ############## RUN Monitor ################################################
    # monitor()
