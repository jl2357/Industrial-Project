import rospy
import subprocess
import time
from nav_msgs.msg import Odometry
from datetime import datetime


class RobotPos:

	# Initialize variables for shutdown of nodes
	roscore_init = None
	jetbotLaunch_init = None	
	

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
		global roscore_init, jetbotLaunch_init
		print ("Shutting Down ROS Nodes")
		
		if jetbotLaunch_init is not None: 
			jetbotLaunch_init.terminate()
			print("Chassis Node Terminated")
	
		if roscore_init is not None:
			roscore_init.terminate()
			print("Master Node Terminated")

	# Get Odometry data
	def odomData(self, msg):
		
		# Get data on robot's postion from /odom
		data = msg.pose.pose.position

		# Display Coordinates
		print("Jetbot Coordinates (x,y,z,):", data.x, data.y, data.z)

		# Write human location data to humanLocData.txt file
		humanLocFile = open("humanLocData.txt", "a")
		humanLocFile.write("\n" + str(datetime.today().strftime('%d/%m/%Y %H:%M:%S')) + " (" + str(round(data.x, 2)) + " " + str(round(data.y, 2)) + ")")

		
	# Get current position of robot
	def getCords(self):
		
		# Initialize ROS node
		rospy.init_node('odometry_listener', anonymous = True)

		# Subscribe to /odom to get data from it
		sub_once = rospy.Subscriber('/odom', Odometry, self.odomData)
		
		# Print data only for 0.1 seconds
		rospy.sleep(0.1)

		# Unregister to exit subscribe loop, otherwise odomData keeps running
		sub_once.unregister()
		
		

