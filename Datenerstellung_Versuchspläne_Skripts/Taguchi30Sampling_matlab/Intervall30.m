% Q (the number of the levels) and N (the number of the factors).
%Create an intervall
Q=30;
N=10;

intervall_x = -320:320;
length_intervall_x = length(intervall_x)-1
abstand_x = length_intervall_x/(Q-1)
intervall_y = -142:142;
length_intervall_y = length(intervall_y)-1
abstand_y = length_intervall_y/(Q-1)
intervall_x = -320:abstand_x:320
intervall_y = -142:abstand_y:142

save('Intervall30.mat',"intervall_x","intervall_y")