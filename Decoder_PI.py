#### MP-VSD ####
import math
import random
import numpy as np
from numpy.linalg import inv , matrix_rank , det
import sympy
from PIL import Image
import socket

#Parity Check Matrix
H = [576460752316072982, 288230376422541329, 144115188094887040, 72057594041729696, 36028797022245126, 18014398543593673, 9007199489885712, 4503600348856329, 2251800082258961, 1125900058275840, 562950558728226, 281475379961988, 140737824980999, 70368796147976, 35184946980872, 17592454517090, 8796161228841, 4398604435522, 2199059448368, 1099679931904, 549919457316, 274877989588, 137984738504, 68865231168, 34426869296, 17469442176, 9139659012, 4630516489, 2699305216, 1149380674]

#Initial Value
Number_Column = 60
Number_Row = 30
Number_bit_data = 32
Number_bit_Y = "#0"+str(Number_bit_data+2)+"b"
Number_bit_H = "#0"+str(Number_Column+2)+"b"

#All possible case of Error Locating Vector
count_select = 2
flag_count_case = 0
count_case_select = 0
select_S = []
Temp_select_S = []
while True:
	if count_select-1 == len(H):
		break
	if flag_count_case == 0:
		for i in range(0,len(H)-count_select+2):
			flag_count_case = flag_count_case+i	
		for i in range(0,count_select):
			Temp_select_S.append(i)
	temp_temp = []
	for i in range(0,len(Temp_select_S)):
		temp_temp.append(Temp_select_S[i])
	select_S.append(temp_temp)
	if count_case_select == flag_count_case-1:
		count_case_select = 0
		flag_count_case = 0
		count_select = count_select+1
		Temp_select_S = []
		continue	
	if Temp_select_S[-1] < len(H)-1:
		Temp_select_S[-1] = Temp_select_S[-1]+1
	else:
		for i in range(0,count_select-1):
			Temp_select_S[-count_select+i] = Temp_select_S[-count_select+i]+1
		Temp_select_S[-1] = Temp_select_S[-2]+1
	count_case_select = count_case_select+1

#Function
def Compute_C(H_matrix,Y_Symbol):
	C_matrix = []
	for i in range(0,Number_Row):
		C_matrix.append(0)
	for i in range(0,len(H_matrix)):
		for j in range(0,len(Y_Symbol)):
			if int(format(H_matrix[i],Number_bit_H)[2+j]) != 0:
				C_matrix[i] = C_matrix[i]^Y_Symbol[j]
	return C_matrix

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

def MatrixBinary_MatrixDec(Binary_matrix,Bits,Number):
	Decimal_matrix = []
	for i in range(0,Number):
		Decimal_matrix.append(0)
		for j in range(0,Bits):
			if int(Binary_matrix[i][j]) == 1:
				Decimal_matrix[i] = Decimal_matrix[i]+pow(2,(Bits-1)-j)
	return Decimal_matrix

def Compute_Verified(H_matrix,C_matrix):
	Decimal_Verified = 0
	for i in range(0,len(C_matrix)):
		if C_matrix[i] == 0:
			Decimal_Verified = H_matrix[i]|Decimal_Verified
	return Decimal_Verified

def Compute_Not_Verified(Y_Symbol,Verified_10):
	Not_Verified = []
	for i in range(0,len(Y_Symbol)):
		if int(format(Verified_10,Number_bit_H)[2+i]) == 0:
			Not_Verified.append((int(format(Verified_10,Number_bit_H)[2+i])+1)*i)
	return Not_Verified

def Compute_Vector_Error(H_matrix,S_matrix,Case_S):
	Decimal_Vector_Error = 0
	for i in range(0,len(S_matrix)):
		if S_matrix[i] == 0:
			Decimal_Vector_Error = Decimal_Vector_Error|H_matrix[i]
	for i in range(0,len(Case_S)):
		for j in range(0,len(Case_S[i])):
			if j == 0:
				sum_s = S_matrix[Case_S[i][j]]
			else:
				sum_s = sum_s^S_matrix[Case_S[i][j]]
		if sum_s == 0:
			for j in range(0,len(Case_S[i])):
				if j == 0:
					sum_h = H_matrix[Case_S[i][j]]
				else:
					sum_h = sum_h^H_matrix[Case_S[i][j]]
			Decimal_Vector_Error = Decimal_Vector_Error|sum_h
	return Decimal_Vector_Error

correct_mp = 0
correct_vsp = 0
symbolerror_total = 0
count_symbol_error = 0
check_channel = 0

height = 1500
width = 1500

receive_data = []
for i in range(0,int(width*height/30)):
	data = 0
	data = conn.recv(1024)
	data = str(data)
	data = data.split(',')
	data = list(data)
	data[0] = data[0].replace('b','')
	data[0] = data[0].replace("'",'')
	data[0] = data[0].replace('[','')
	data[-1] = data[-1].replace("'",'')
	data[-1] = data[-1].replace("]",'')
	receive_data.append(data)
	#print(data)
	flag = "Finish"
	reply = flag.encode()
	conn.send(reply)

decode_data = []
pre_correct = []

for channel in range(0,int(width*height/30)):
###############################MP################################
	print(channel)
	Y = []
	for i in range(0,Number_Column):
		Y.append(int(receive_data[channel][i]))

	E_After_Process = []
	for i in range(0,Number_Column):
		E_After_Process.append(0)
	
	C = Compute_C(H,Y)
	Verified = Compute_Verified(H,C)
	Number_Y_NOTVerified = Compute_Not_Verified(Y,Verified)
	
	i = 0
	count = 0
	check_node = 0
	flag_duplicate = 0
	C_Duplicate = []
	C_Duplicate_Count = []
	C_Duplicate_Number = []
	Y_Duplicate = []
	Y_Duplicate_Sort = []
	Y_Duplicate_Count = []
	
	while True:
		count = 0
		if i > len(C)-1 :
			if flag_duplicate == 0:
				flag_duplicate = 1
				i = 0
				continue
			break
		else :
			if flag_duplicate == 1 and check_node == 0:
				Count_Duplicate = 0
				for num in C:
					if num not in C_Duplicate:
						C_Duplicate.append(num)
				for num in range(0,len(C_Duplicate)):
					for num_1 in range(0,len(C)):
						if int(C[num_1]) == int(C_Duplicate[num]) :
							 Count_Duplicate = Count_Duplicate+1
					C_Duplicate_Count.append(Count_Duplicate)
					Count_Duplicate = 0
				flag_check_duplicate = 0
				for num in range(0,len(C_Duplicate)):
					if int(C_Duplicate_Count[num])>1 and int(C_Duplicate[num]) != 0 :
						flag_check_duplicate = 1
				if flag_check_duplicate == 0 :
					break
				else:
					num = 0
					while True:
						Y_Duplicate_Count = []
						if num == int(len(C_Duplicate)):
							break
						if C_Duplicate[num] == 0:
							num = num+1
							continue
						else:
							if C_Duplicate_Count[num] > 1:
								if C_Duplicate_Count[num] == 2:
									y = -1
									Y_Duplicate = []
									C_Duplicate_Number = []
									flag_y = -1
									for num_11 in range(0,len(C)):
										if int(C[num_11]) == int(C_Duplicate[num]):
											C_Duplicate_Number.append(num_11)
											if flag_y == -1:
												flag_y = num_11
									for num_1 in C_Duplicate_Number:
										C_Composition = Decimal_Binary(H[num_1],Number_bit_H)
										for num_2 in range(0,len(C_Composition)):
											for num_3 in range(0,len(Number_Y_NOTVerified)):
												if int(C_Composition[num_2])==1 and int(Number_Y_NOTVerified[num_3]) == num_2:
													Y_Duplicate.append(num_2)
									Count_Duplicate = 0
									for num_1 in Y_Duplicate:
										if num_1 not in Y_Duplicate_Sort:
											Y_Duplicate_Sort.append(num_1)
									for num_4 in range(0,len(Y_Duplicate_Sort)):
										for num_1 in range(0,len(Y_Duplicate)):
											if Y_Duplicate[num_1] == Y_Duplicate_Sort[num_4] :
												Count_Duplicate = Count_Duplicate+1
										Y_Duplicate_Count.append(Count_Duplicate)
										Count_Duplicate = 0
									flag_Y_Dup = 0
									for num_1 in Y_Duplicate_Count:
										if int(num_1)>1:
											flag_Y_Dup = flag_Y_Dup+1
									if flag_Y_Dup == 1:
										for num_1 in range(0,len(Y_Duplicate_Sort)):
											if Y_Duplicate_Count[num_1] > 1:
												y = Y_Duplicate_Sort[num_1]
												i = 0
												break
								else:
									fix_case = 0
									count_fix_case = int(math.factorial(int(C_Duplicate_Count[num]))/(math.factorial(2)*math.factorial(int(C_Duplicate_Count[num])-2))) 
									Select_Case = []
									for num_1 in range(0,int(C_Duplicate_Count[num])):
										Select_Case.append(num_1+1)
									Select_Case_Pair = []
									for pair1 in range(0,int(C_Duplicate_Count[num])-1):
										for pair2 in range(pair1+1,int(C_Duplicate_Count[num])):
											Select_Case_Pair.append(pair1)
											Select_Case_Pair.append(pair2)
									pair1 = 0
									pair2 = 1
									C_Duplicate_Number = []
									for num_1 in range(0,len(C)):
										if int(C[num_1]) == int(C_Duplicate[num]):
											C_Duplicate_Number.append(num_1)																	
									while True:
										if fix_case == count_fix_case:
											break
										fix_case = fix_case+1
										y = -1
										Y_Duplicate = []
										Y_Duplicate_Count = []
										C_Duplicate_Number_fix = []
										C_Duplicate_Number_fix.append(C_Duplicate_Number[int(Select_Case_Pair[pair1])])
										C_Duplicate_Number_fix.append(C_Duplicate_Number[int(Select_Case_Pair[pair2])])
										pair1 = pair1+2
										pair2 = pair2+2
										for num_1 in C_Duplicate_Number_fix:
											C_Composition = Decimal_Binary(H[num_1],Number_bit_H)
											for num_2 in range(0,len(C_Composition)):
												for num_3 in range(0,len(Number_Y_NOTVerified)):
													if int(C_Composition[num_2])==1 and int(Number_Y_NOTVerified[num_3]) == num_2:
														Y_Duplicate.append(num_2)
										Count_Duplicate = 0
										for num_1 in Y_Duplicate:
											if num_1 not in Y_Duplicate_Sort:
												Y_Duplicate_Sort.append(num_1)
										for num_4 in range(0,len(Y_Duplicate_Sort)):
											for num_1 in range(0,len(Y_Duplicate)):
												if Y_Duplicate[num_1] == Y_Duplicate_Sort[num_4] :
													Count_Duplicate = Count_Duplicate+1
											Y_Duplicate_Count.append(Count_Duplicate)
											Count_Duplicate = 0
										flag_Y_Dup = 0
										for num_1 in Y_Duplicate_Count:
											if int(num_1)>1:
												flag_Y_Dup = flag_Y_Dup+1
										if flag_Y_Dup == 1:
											for num_1 in range(0,len(Y_Duplicate_Sort)):
												if Y_Duplicate_Count[num_1] > 1:
													y = Y_Duplicate_Sort[num_1]
													i = 0
													break
										if y != -1:								
											break
								if y == -1:
									num = num+1
									continue
								else:
									if C_Duplicate_Count[num] == 2 :
										Y[y] = Y[y]^C[flag_y]									
									else:
										Y[y] = Y[y]^C[C_Duplicate_Number[int(Select_Case_Pair[pair1-2])]]
									C = Compute_C(H,Y)
									Verified = Compute_Verified(H,C)
									Number_Y_NOTVerified = Compute_Not_Verified(Y,Verified)
									Count_Duplicate = 0
									C_Duplicate = []
									C_Duplicate_Count = []
									for num_1 in C:
										if num_1 not in C_Duplicate:
											C_Duplicate.append(num_1)
									for num_2 in range(0,len(C_Duplicate)):
										for num_1 in range(0,len(C)):
											if int(C[num_1]) == int(C_Duplicate[num_2]) :
												 Count_Duplicate = Count_Duplicate+1
										C_Duplicate_Count.append(Count_Duplicate)
										Count_Duplicate = 0
									flag_check_duplicate = 0
									for num_1 in range(0,len(C_Duplicate)):
										if int(C_Duplicate_Count[num_1])>1 and int(C_Duplicate[num_1]) != 0 :
											flag_check_duplicate = 1
									if flag_check_duplicate == 0 :
											break
									num = 0
									continue 
						num = num+1	
					check_node = 1
					i = 0			
			C_Composition = Decimal_Binary(H[i],Number_bit_H)
			flag = 0
			for j in range(0,len(C_Composition)):
				if int(C_Composition[j]) == 1 :
					for x in range(0,len(Number_Y_NOTVerified)):
						if int(Number_Y_NOTVerified[x]) == j:
							flag = 1
							break
				if flag == 1:
					count = count+1
					if count > 1:
						break
					y = j
				flag = 0
					
			if count != 1:
				i = i+1
				continue
			else:
				Y[y] = Y[y]^C[i]
				C = Compute_C(H,Y)
				Verified = Compute_Verified(H,C)
				Number_Y_NOTVerified = Compute_Not_Verified(Y,Verified)	
				i = 0
		
	C = Compute_C(H,Y)
	
	flag_Can_Correct = 1
	for i in range(0,len(C)):
		if C[i] != 0:
			flag_Can_Correct = 0
			break
	
	if flag_Can_Correct == 1:
		correct_mp = correct_mp+1
	#	case_error_can_correct_Systematic_MP[percolumn][ex][count_error] = int(case_error_can_correct_Systematic_MP[percolumn][ex][count_error])+1
	#	count_correct_Systematic[percolumn][ex] = count_correct_Systematic[percolumn][ex]+1
#########################Can't Correct By MP########################
	if flag_Can_Correct == 0:
		H_binary = MatrixDec_MatrixBinary(H,Number_bit_H)
		Y_binary = MatrixDec_MatrixBinary(Y,Number_bit_Y)
		H_binary = np.array(H_binary,'int')
		Y_binary = np.array(Y_binary,'int')
		S_Binary = np.mod(np.dot(H_binary,Y_binary),2)
		S_Binary = np.floor(S_Binary)
		S = MatrixBinary_MatrixDec(S_Binary,Number_bit_data,Number_Row)

		#Compute Vector Error
		Vector_Error = Compute_Vector_Error(H,S,select_S)
		Vector_Error_Binary = Decimal_Binary(Vector_Error,Number_bit_H)
		count_vector_error = 0
		for i in range(0,len(Vector_Error_Binary)):
			if int(Vector_Error_Binary[i]) == 0:
				count_vector_error = count_vector_error+1

###############################VSD##################################
		S_sub = []
		S_sub_position = []
		Vector_Error_Binary_position = []
		count_S_sub = 0
		flag_count_S_sub = 0
	
		for i in range(0,len(Vector_Error_Binary)):
			if int(Vector_Error_Binary[i]) == 0:
				flag_count_S_sub = flag_count_S_sub+1
				Vector_Error_Binary_position.append(i)
				
		if len(Vector_Error_Binary_position) <= len(S):
			S_check = np.array(S_Binary)
			rref_s = 0
			inds = 0
			rref_s, inds = sympy.Matrix(S_check).T.rref()
			error_data = []
			inds = list(inds)
			for i in range(0,len(Vector_Error_Binary_position)):
				if int(Vector_Error_Binary_position[i]) < Number_Row:
					error_data.append(Vector_Error_Binary_position[i])
			for i in range(0,len(error_data)):
				if error_data[i] in inds:
					inds.remove(int(error_data[i]))				
			
			#All possible case					
			count_select = flag_count_S_sub-len(error_data)
			if count_select < len(inds):
				Set_inds = []
				for i in range(0,len(inds)):
					Set_inds.append(i)
				Temp_select_S_sub = []
				case_select_S_sub = []
				count_case_select = 0
				flag_count_case = 0	
				while True:
					if flag_count_case == 0:
						for i in range(0,len(Set_inds)-count_select+2):
							flag_count_case = flag_count_case+i	
						for i in range(0,count_select):
							Temp_select_S_sub.append(i)
					temp_temp = []
					for i in range(0,len(Temp_select_S_sub)):
						temp_temp.append(Temp_select_S_sub[i])
					case_select_S_sub.append(temp_temp)
					if count_case_select == flag_count_case-1:
						break	
					if Temp_select_S_sub[-1] < len(Set_inds)-1:
						Temp_select_S_sub[-1] = Temp_select_S_sub[-1]+1
					else:
						for i in range(0,count_select-1):
							Temp_select_S_sub[-count_select+i] = Temp_select_S_sub[-count_select+i]+1
						Temp_select_S_sub[-1] = Temp_select_S_sub[-2]+1
					count_case_select = count_case_select+1
				for i in range(0,len(case_select_S_sub)):
					select_S_sub = []
					for j in range(0,len(error_data)):
						select_S_sub.append(error_data[j])
					for j in range(0,len(case_select_S_sub[i])):
						select_S_sub.append(inds[case_select_S_sub[i][j]])
	
					S_sub = []
					S_sub_position = []
					for j in range(0,len(select_S_sub)):
						S_sub.append(S_Binary[select_S_sub[j]])
						S_sub_position.append(select_S_sub[j])	
					
					H_sub = []
					if len(S_sub_position) == len(Vector_Error_Binary_position):
						for i in range(0,len(S_sub_position)):
							Temp_H_sub = []
							for j in range(0,len(Vector_Error_Binary_position)):
								Temp_H_sub.append(H_binary[int(S_sub_position[i])][int(Vector_Error_Binary_position[j])])
							H_sub.append(Temp_H_sub)
						H_sub_np = np.array(H_sub)
						H_sub_inverse_np = np.array(H_sub)
						S_sub = np.array(S_sub)
	
						if det(H_sub_np) == 0:
							continue
						else:
							H_sub_inverse_np = inv(H_sub_inverse_np)
							H_sub_inverse_np = np.mod(H_sub_inverse_np,2)
							H_sub_inverse_np = np.floor(H_sub_inverse_np)
							H_sub_S_sub_binary = np.dot(H_sub_inverse_np,S_sub)
							H_sub_S_sub_binary = np.mod(H_sub_S_sub_binary,2)
							H_sub_S_sub_binary = np.floor(H_sub_S_sub_binary)
							H_sub_S_sub = []
							for i in range(0,len(S_sub_position)):
								H_sub_S_sub.append(0)
								for j in range(0,Number_bit_data):
									if int(H_sub_S_sub_binary[i][j]) == 1:
										H_sub_S_sub[i] = H_sub_S_sub[i]+pow(2,(Number_bit_data-1)-j)
	
							for i in range(0,len(Vector_Error_Binary_position)):
								E_After_Process[Vector_Error_Binary_position[i]] = H_sub_S_sub[i]		
							
							Temp_Y = []
							for i in range(0,len(Y)):
								Temp_Y.append(0)
								Value_Y = 0
								Value_Y = Y[i]
								Temp_Y[i] = Value_Y
							
							for i in range(0,len(Temp_Y)):
								Temp_Y[i] = Temp_Y[i]^E_After_Process[i]	
	
							Temp_Y_binary = MatrixDec_MatrixBinary(Temp_Y,Number_bit_Y)	
							Temp_Y_binary = np.array(Temp_Y_binary,'int')
							S_Binary = np.mod(np.dot(H_binary,Temp_Y_binary),2)
							S = MatrixBinary_MatrixDec(S_Binary,Number_bit_data,Number_Row)
							Vector_Error = Compute_Vector_Error(H,S,select_S)
							Vector_Error_Binary = Decimal_Binary(Vector_Error,Number_bit_H)
							if 0 in Vector_Error_Binary:
								for i in range(0,len(Vector_Error_Binary_position)):
									E_After_Process[Vector_Error_Binary_position[i]] = 0
								continue
							else:
								for i in range(0,len(Temp_Y)):
									Y[i] = Y[i]^E_After_Process[i]
	
								C = Compute_C(H,Y)
								
								flag_Can_Correct = 1
								for i in range(0,len(C)):
									if C[i] != 0:
										flag_Can_Correct = 0
										break
								
								if flag_Can_Correct == 1:
									correct_vsp = correct_vsp+1
									#case_error_can_correct_Systematic_VSD[percolumn][ex][count_error] = int(case_error_can_correct_Systematic_VSD[percolumn][ex][count_error])+1
									#count_correct_Systematic[percolumn][ex] = count_correct_Systematic[percolumn][ex]+1
									break
		for i in range(30,len(Vector_Error_Binary)):
			if int(Vector_Error_Binary[i]) == 0:
				count_symbol_error = count_symbol_error+1

	Data_decoder = []
	pre_correct_temp = []
	for i in range(30,Number_Column):
		Data_decoder.append(Y[i])
		pre_correct_temp.append(int(receive_data[channel][i]))
	decode_data.append(Data_decoder)	
	pre_correct.append(pre_correct_temp)
						
data_color = []
for i in range(0,len(pre_correct)):
	for j in range(0,len(pre_correct[i])):
		color = [0,0,0]
		red = 0
		green = 0
		blue = 0
		color[0] = (pre_correct[i][j]>>16)&0x0ff
		color[1] = (pre_correct[i][j]>>8)&0x0ff
		color[2] = (pre_correct[i][j])&0x0ff
		temp_color = 0
		temp_color = color
		data_color.append(temp_color)

decode_image = []
for i in range(0,height):
	temp_data = []
	for j in range(0,width):
		temp_data.append(data_color[(i*width)+j])
	decode_image.append(temp_data)
decode_image = np.array(decode_image)
im_pre = Image.fromarray(decode_image.astype('uint8'),"RGB")
im_pre = im_pre.save("pre.jpg")

		
data_color = []
for i in range(0,len(decode_data)):
	for j in range(0,len(decode_data[i])):
		color = [0,0,0]
		red = 0
		green = 0
		blue = 0
		color[0] = (decode_data[i][j]>>16)&0x0ff
		color[1] = (decode_data[i][j]>>8)&0x0ff
		color[2] = (decode_data[i][j])&0x0ff
		temp_color = 0
		temp_color = color
		data_color.append(temp_color)

decode_image = []
for i in range(0,height):
	temp_data = []
	for j in range(0,width):
		temp_data.append(data_color[(i*width)+j])
	decode_image.append(temp_data)
decode_image = np.array(decode_image)
im_after = Image.fromarray(decode_image.astype('uint8'),"RGB")
im_after = im_after.save("after.jpg")


