import socket
import sys
import struct

#Create a TCP/IP socket
def connect_To_Server():
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	#Connect to JAVA Server running at port 9959
	server_address = ('localhost', 9958)
	print >> sys.stdout, 'connecting to %s port %s' % server_address
	sock.connect(server_address)
	sendQueryToServer("Hello Server")

#Method to send query to java server
def sendQueryToServer(detected_gesture):
	try:
		#Sending Query Packet String
		message = gesture_Detected(detected_gesture)
		size = len(message)
		print >>sys.stdout, 'sending "%s"' % message
		#sock.send(struct.pack("!H",size))
		sock.send(message)
	except:
		print >> sys.stdout, 'Failed to send the query to the server!!'




#Function list to convert detected gesture into query.
	#Detected Gesture List : Action to Perform
	#1. Open Hand => Toast to User
	#2. Closed Hand => Toggle play/pause the Video
	#3. Fist => Pausing the Application
	#4. Pointing - X => Back Action
	#5. Pointing - Y => Scroll Down
	#6. Metal => Full Volume
	#7. Gun => Scroll up
	#8. Two => Increase Volume
	#9. Three => Decrease Volume
	#10. Four => Mute

#1. Open Hand
def openHand():
	return "::alert::tester::"

#2. Closed Hand
def closedHand():
	return "::toggle::tester::"

#3. Fist
def fist():
	return "::toggle::tester::"

#4. Pointing - X
def pointing_x():
	return "::back::tester::"

#5. Pointing - Y
def pointing_y():
	return "::scroll down::tester::"

#6. Metal
def metal():
	return "::full volume::tester::"

#7. Gun
def gun():
	return "::scroll up::tester::"

#8. Two
def two():
	return "::increase volume::tester::"

#9. Three
def three():
	return "::decrease volume::tester::"

#10. Four
def four():
	return "::mute::tester::"


#Function to call whenever a Gesture is Detected. The Function takes a String argument defining the detected gesture
#and returns a query linked to the detected gesture.

def gesture_Detected(detected_gesture):
	if detected_gesture == "open-hand":
		return openHand()
	elif detected_gesture == "closed-hand":
		return closedHand()
	elif detected_gesture == "fist":
		return fist()
	elif detected_gesture == "pointing-x":
		return pointing_x()
	elif detected_gesture == "pointing-y":
		return pointing_y()
	elif detected_gesture == "metal":
		return metal()
	elif detected_gesture == "gun":
		return gun()
	elif detected_gesture == "two":
		return two()
	elif detected_gesture == "three":
		return three()
	elif detected_gesture == "four":
		return four()
	elif detected_gesture == "Finish It!":
		return "Finish It!"
	elif detected_gesture() == "Hello Server":
		return "Hello Server"
	else:
		return "error"