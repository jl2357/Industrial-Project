import JetsonCords
import time
from nav_msgs.msg import Odometry

ShutDown = JetsonCords.RobotPos()

ShutDown.odomData(Odometry)
time.sleep(3)
ShutDown.shutNodes()
