import rospy
import subprocess
import time
import math
from nav_msgs.msg import Odometry
from datetime import datetime


class RobotPos:

	# Initialize variables for shutdown of nodes
	roscore_init = None
	jetbotLaunch_init = None	
	personID = None

	# Start Robot Master Node & Robot Chassis Node
	def __init__(self):
		global roscore_init, jetbotLaunch_init
		print ("Initialize Robot Master Node...")
		
		# Run ros command 
		roscore_init = subprocess.Popen(["roscore"])

		# Delay for initialization
		time.sleep(5)

		print("Initializing Robot Chassis Node...")
		
		# Run ros command
		jetbotLaunch_init = subprocess.Popen(["roslaunch", "jetbot_pro", "jetbot.launch"])

		# Delay for initialization
		time.sleep(10)
		

	def shutNodes(self):
		# Shutting down nodes after use
		global roscore_init, jetbotLaunch_init
		print ("Shutting Down ROS Nodes")
		
		if jetbotLaunch_init is not None: 
			jetbotLaunch_init.terminate()
			print("Chassis Node Terminated")
	
		if roscore_init is not None:
			roscore_init.terminate()
			print("Master Node Terminated")

	def eulerYaw_from_quaternion(self, x, y, z, w):
		# Convert quaternions from Odometer orientation data (x,y,z,w) to Euler angles (roll, pitch, yaw)
		# where yaw represents that how much the robot has turned
		
		# Equation to calculate yaw 
		t3 = 2.0 * (w * z + x * y)
		t4 = 1.0 - 2.0 * (y * y + z * z)
		yaw = math.atan2(t3, t4) 
		return yaw

	# Get Odometry data
	def odomData(self, msg):
		
		# Get data on robot's postion from /odom
		pos = msg.pose.pose.position

		# Get data on robot's orientation from /odom
		orien = msg.pose.pose.orientation

		# Use eulerYaw_from_quaternion to get yaw of robot
		yaw = self.eulerYaw_from_quaternion(orien.x, orien.y, orien.z, orien.w)

		# Display Coordinates
		print("Jetbot Coordinates (x,y): ", pos.x, pos.y)
		print("Jetbot Orientation (radians): ", yaw)

		# Write human location data to humanLocData.txt file
		humanLocFile = open("humanLocData.txt", "a")
		humanLocFile.write("\n" + self.personID[:-4] + " " + str(datetime.today().strftime('%d/%m/%Y %H:%M:%S')) + " (" + str(round(pos.x, 6)) + " " + str(round(pos.y, 6)) + ") " + str(round(yaw, 6)))

		
	# Get current position of robot
	def getCords(self, personID):
		
		self.personID = personID
		# Initialize ROS node
		rospy.init_node('odometry_listener', anonymous = True)

		# Subscribe to /odom to get data from it
		sub_once = rospy.Subscriber('/odom', Odometry, self.odomData)
		
		# Print data only for 0.1 seconds
		rospy.sleep(0.1)

		# Unregister to exit subscribe loop, otherwise odomData keeps running
		sub_once.unregister()
		
		

