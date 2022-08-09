import random
from PIL import Image
import socket
import numpy as np

def Decimal_Binary(Decimal,Bits):
	Binary = [x for x in format(Decimal,Bits)]
	del Binary[0]
	del Binary[0]
	return Binary

def MatrixDec_MatrixBinary(Decimal_matrix,Bits):
	Binary_matrix = []
	for i in range(0,len(Decimal_matrix)):
		Composition = Decimal_Binary(Decimal_matrix[i],Bits)
		Binary_matrix.append(Composition)
	return Binary_matrix

G = [5106280712634368, 290922002359779328, 14919891886800896, 9719722585096192, 31634049076297728, 149820021353218048, 576461520037216256, 576500912399319040, 540434225176772608, 108869312056393728, 90427272242659328, 298961620376485888, 255579297107017728, 581321418331979776, 307537817515360256, 144693840429727744, 145557759142666240, 39425227091873792, 290518546822203392, 731836210758026240, 81068405360165376, 36116841701048576, 234469085096706176, 18036870852116544, 72649715409223712, 875952635784462352, 22632489080061960, 612912595011108868, 613215228070461442, 313149711978594305]				

#Initial Value
Number_Column = 60
Number_Row = 30
Number_bit_data = 32
Number_bit_G = "#0"+str(Number_Column+2)+"b"

G_binary = MatrixDec_MatrixBinary(G,Number_bit_G)

check_channel = 0
height = 1500
width = 1500

q = 0.9
p = 0.05

im = Image.open("ku.png")
im = np.array(im)
data_im = []
for i in range(0,len(im)):
	for j in range(0,len(im[i])):
		rgb = 0
		rgb = ((im[i][j][0]&0x0ff)<<16)|((im[i][j][1]&0x0ff)<<8)|(im[i][j][2]&0x0ff)
		data_im.append(rgb)
arr_data = []
for i in range(0,int(len(data_im)/Number_Row)):
	temp_data = []
	for j in range(0,Number_Row):
		temp_data.append(data_im[(i*Number_Row)+j])
	arr_data.append(temp_data)

Encoded_data = []

for number in range(0,len(arr_data)):
	V = []
	for i in range(0,Number_Column):
		V.append(0)
	for i in range(0,Number_Column):
		for j in range(0,Number_Row):
			if int(G_binary[j][i]) == 1:
				V[i] = V[i]^arr_data[number][j]	
	Y = []
	E = []

	for i in range(0,Number_Column):
		Y.append(0)			
		E.append(0)

	Select_matrix = []
	for num in range(0,Number_Column):
	 if check_channel == 0:
	 	random_uniform = random.uniform(0,1)
	 	if random_uniform < 1-p:
	 		check_channel = 0
	 	else:
	 		check_channel = 1
	 else:
	 	random_uniform = random.uniform(0,1)
	 	if random_uniform < q:
	 		check_channel = 0
	 	else:
	 		check_channel = 1
	 Select_matrix.append(check_channel)
	
	for i in range(0,Number_Column):
		if int(Select_matrix[i]) != 0:
			E[i] = random.randint(1,pow(2,32)-1)

	for i in range(0,len(Y)):
		Y[i] = V[i]^E[i]

	Encoded_data.append(Y)

print(Encoded_data)

#Connect TCP
HOST = '192.168.43.175' # Enter IP or Hostname of your server
PORT = 12345 # Pick an open Port (1000+ recommended), must match the server port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST,PORT))
for i in range(0,len(Encoded_data)):	
	sending = str(Encoded_data[i])
	sending = sending.encode()
	s.send(sending)
	received = s.recv(1024)
s.close()