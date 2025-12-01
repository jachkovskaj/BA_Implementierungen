% load('matlab.mat','Knoten','Elemente','Elementenanzahl');
% % load('Elementenzahl');
% % load('Knoten');
% % load('Elemente.mat');
% %load ('Gruppen_Nummer_Schleife.mat')
% load('input.mat')
% n=10;
% %% Erstellen des Taguchi Versuchsplans "ParameterArray"
% load('TaguchiArray.mat')
% load('Intervalle.mat')
% ParameterArray=TaguchiArray;
% Intervalldesignvariablen=0:(250/(n-1)):250;
% V = randperm(n);
% Intervalldesignvariable1=Intervalldesignvariablen(V);
% V = randperm(n);
% Intervalldesignvariable2=Intervalldesignvariablen(V);
% V = randperm(n);
% Intervalldesignvariable3=Intervalldesignvariablen(V);
% V = randperm(n);
% Intervalldesignvariable4=Intervalldesignvariablen(V);
% V = randperm(n);
% Intervalldesignvariable5=Intervalldesignvariablen(V);
% V = randperm(n);
% Intervalldesignvariable6=Intervalldesignvariablen(V);
% V = randperm(n);
% Intervalldesignvariable7=Intervalldesignvariablen(V);
% V = randperm(n);
% Intervalldesignvariable8=Intervalldesignvariablen(V);
% Intervallpositionx=0:(200/(n-1)):200;
% Intervallpositiony=-75:(75/(n-1)):0;
% 
% 
%     for k=1:1:(n^2)
%        ParameterArray(k,3)=Intervalldesignvariable1(ParameterArray(k,3));
%        ParameterArray(k,4)=Intervalldesignvariable2(ParameterArray(k,4));
%        ParameterArray(k,5)=Intervalldesignvariable3(ParameterArray(k,5));
%        ParameterArray(k,6)=Intervalldesignvariable4(ParameterArray(k,6));
%        ParameterArray(k,7)=Intervalldesignvariable5(ParameterArray(k,7));
%        ParameterArray(k,8)=Intervalldesignvariable6(ParameterArray(k,8));
%        ParameterArray(k,9)=Intervalldesignvariable7(ParameterArray(k,9));
%        ParameterArray(k,10)=Intervalldesignvariable8(ParameterArray(k,10));
%        ParameterArray(k,1)=Intervallpositionx(ParameterArray(k,1));
%        ParameterArray(k,2)=Intervallpositiony(ParameterArray(k,2));
%     end    
% 
% 
% designvariablen_aktuell=zeros(8);
% Hight=zeros(n^2,0);
% 
%% Erstellen der Gex nach dem Versuchsplan
for i=1:1:49%n^2
   
    positionx_aktuell=ParameterArray(i,1);
    positiony_aktuell=ParameterArray(i,2);
    designvariablen_aktuell=ParameterArray(i,3:10);
    [Charge,Charge_komplett]=variabletransformation(positionx_aktuell, positiony_aktuell, designvariablen_aktuell, Elemente, Elementenanzahl, Knoten, [],i);
    Area=sum(Charge_komplett);
    Einleger=zeros(1,1);
    Hight(i)=(6503/Area)*3;
    x=1;
    y=0;
    for j=1:1:Elementenanzahl
        if Charge_komplett(j,1)==1
        y=y+1;
        Einleger(x,y)=j;
         if y==4
         y=0;
         x=x+1;
        end
        end
    end

filename='Platte_Taguchi10^2-1.gex';
fileID=fopen(filename,'a');
fprintf(fileID, '%s \n', "Gruppeninformation:");
fprintf(fileID,'%s, %s, %s \n',"Gruppen-Nr.", "Anzahl der Elemente", "Gruppennamme:");
fprintf(fileID, '%10i %10i \n',i, Area);
fprintf(fileID, '%i \n', i);
fprintf(fileID, '%s \n', "Elemente:");

Einleger=cast(Einleger,'int16');
fileID=fopen(filename,'a');
fprintf(fileID,'%10i %9i %9i %9i\n',Einleger);

end
%% Erstellen der Gex nach dem Versuchsplan
for i=50:1:98%n^2
   
    positionx_aktuell=ParameterArray(i,1);
    positiony_aktuell=ParameterArray(i,2);
    designvariablen_aktuell=ParameterArray(i,3:10);
    [Charge,Charge_komplett]=variabletransformation(positionx_aktuell, positiony_aktuell, designvariablen_aktuell, Elemente, Elementenanzahl, Knoten, [],i);
    Area=sum(Charge_komplett);
    Einleger=zeros(1,1);
    Hight(i)=(6503/Area)*3;
    x=1;
    y=0;
    for j=1:1:Elementenanzahl
        if Charge_komplett(j,1)==1
        y=y+1;
        Einleger(x,y)=j;
         if y==4
         y=0;
         x=x+1;
        end
        end
    end

filename='Platte_Taguchi10^2-2.gex';
fileID=fopen(filename,'a');
fprintf(fileID, '%s \n', "Gruppeninformation:");
fprintf(fileID,'%s, %s, %s \n',"Gruppen-Nr.", "Anzahl der Elemente", "Gruppennamme:");
fprintf(fileID, '%10i %10i \n',i, Area);
fprintf(fileID, '%i \n', i);
fprintf(fileID, '%s \n', "Elemente:");

Einleger=cast(Einleger,'int16');
fileID=fopen(filename,'a');
fprintf(fileID,'%10i %9i %9i %9i\n',Einleger);


end
%% Erstellen der Gex nach dem Versuchsplan
for i=99:1:100%n^2
   
    positionx_aktuell=ParameterArray(i,1);
    positiony_aktuell=ParameterArray(i,2);
    designvariablen_aktuell=ParameterArray(i,3:10);
    [Charge,Charge_komplett]=variabletransformation(positionx_aktuell, positiony_aktuell, designvariablen_aktuell, Elemente, Elementenanzahl, Knoten, [],i);
    Area=sum(Charge_komplett);
    Einleger=zeros(1,1);
    Hight(i)=(6503/Area)*3;
    x=1;
    y=0;
    for j=1:1:Elementenanzahl
        if Charge_komplett(j,1)==1
        y=y+1;
        Einleger(x,y)=j;
         if y==4
         y=0;
         x=x+1;
        end
        end
    end

filename='Platte_Taguchi10^2-3.gex';
fileID=fopen(filename,'a');
fprintf(fileID, '%s \n', "Gruppeninformation:");
fprintf(fileID,'%s, %s, %s \n',"Gruppen-Nr.", "Anzahl der Elemente", "Gruppennamme:");
fprintf(fileID, '%10i %10i \n',i, Area);
fprintf(fileID, '%i \n', i);
fprintf(fileID, '%s \n', "Elemente:");

Einleger=cast(Einleger,'int16');
fileID=fopen(filename,'a');
fprintf(fileID,'%10i %9i %9i %9i\n',Einleger);


end
save('Trainingsdaten10^2/Metadaten.mat');
