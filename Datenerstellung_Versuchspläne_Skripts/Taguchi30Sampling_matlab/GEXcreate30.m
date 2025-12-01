load('07_09_2021_Workspace_lm_Bayes_Biegung_3_5.mat','network_array','Knoten','Elemente','Elementenanzahl');
% load('Elementenzahl');
% load('Knoten');
% load('Elemente.mat');
%load ('Gruppen_Nummer_Schleife.mat')
load('Loeschliste.mat')
load('input.mat')
n=37; %Stufen
%% Erstellen des Taguchi Versuchsplans "ParameterArray"
load('Taguchi30.mat')
load('Intervall30.mat')
ParameterArray=Ta;
Intervalldesignvariablen=10:(700/(n-1)):700;

    for k=1:1:(length(Ta))
       ParameterArray(k,3)=Intervalldesignvariablen(ParameterArray(k,3));
       ParameterArray(k,4)=Intervalldesignvariablen(ParameterArray(k,4));
       ParameterArray(k,5)=Intervalldesignvariablen(ParameterArray(k,5));
       ParameterArray(k,6)=Intervalldesignvariablen(ParameterArray(k,6));
       ParameterArray(k,7)=Intervalldesignvariablen(ParameterArray(k,7));
       ParameterArray(k,8)=Intervalldesignvariablen(ParameterArray(k,8));
       ParameterArray(k,9)=Intervalldesignvariablen(ParameterArray(k,9));
       ParameterArray(k,10)=Intervalldesignvariablen(ParameterArray(k,10));
       ParameterArray(k,1)=intervall_x(ParameterArray(k,1));
       ParameterArray(k,2)=intervall_y(ParameterArray(k,2));
    end    

designvariablen_aktuell=zeros(8);
Hight=zeros(length(Ta),1);

%% Erstellen der Gex nach dem Versuchsplan
for i=1:1:49 %n^2
   
    positionx_aktuell=ParameterArray(i,1);
    positiony_aktuell=ParameterArray(i,2);
    designvariablen_aktuell=ParameterArray(i,3:10);
    [Charge,Charge_komplett]=variabletransformation(positionx_aktuell, positiony_aktuell, designvariablen_aktuell, Elemente, Elementenanzahl, Knoten, Loeschliste,i);
    Area=sum(Charge_komplett);
    Einleger=zeros(1,1);
    Hight(i)=(6503/Area)*3;
    x=1;
    y=0;
    for j=1:1:6503
        if Charge_komplett(j,1)==1
        y=y+1;
        Einleger(x,y)=j;
         if y==4
         y=0;
         x=x+1;
        end
        end
    end

filename='Taguchi30-1.gex';
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
for i=50:1:98 %n^2
   
    positionx_aktuell=ParameterArray(i,1);
    positiony_aktuell=ParameterArray(i,2);
    designvariablen_aktuell=ParameterArray(i,3:10);
    [Charge,Charge_komplett]=variabletransformation(positionx_aktuell, positiony_aktuell, designvariablen_aktuell, Elemente, Elementenanzahl, Knoten, Loeschliste,i);
    Area=sum(Charge_komplett);
    Einleger=zeros(1,1);
    Hight(i)=(6503/Area)*3;
    x=1;
    y=0;
    for j=1:1:6503
        if Charge_komplett(j,1)==1
        y=y+1;
        Einleger(x,y)=j;
         if y==4
         y=0;
         x=x+1;
        end
        end
    end

filename='Taguchi30-2.gex';
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
for i=99:1:147 %n^2
   
    positionx_aktuell=ParameterArray(i,1);
    positiony_aktuell=ParameterArray(i,2);
    designvariablen_aktuell=ParameterArray(i,3:10);
    [Charge,Charge_komplett]=variabletransformation(positionx_aktuell, positiony_aktuell, designvariablen_aktuell, Elemente, Elementenanzahl, Knoten, Loeschliste,i);
    Area=sum(Charge_komplett);
    Einleger=zeros(1,1);
    Hight(i)=(6503/Area)*3;
    x=1;
    y=0;
    for j=1:1:6503
        if Charge_komplett(j,1)==1
        y=y+1;
        Einleger(x,y)=j;
         if y==4
         y=0;
         x=x+1;
        end
        end
    end

filename='Taguchi30-3.gex';
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
for i=148:1:196 %n^2
   
    positionx_aktuell=ParameterArray(i,1);
    positiony_aktuell=ParameterArray(i,2);
    designvariablen_aktuell=ParameterArray(i,3:10);
    [Charge,Charge_komplett]=variabletransformation(positionx_aktuell, positiony_aktuell, designvariablen_aktuell, Elemente, Elementenanzahl, Knoten, Loeschliste,i);
    Area=sum(Charge_komplett);
    Einleger=zeros(1,1);
    Hight(i)=(6503/Area)*3;
    x=1;
    y=0;
    for j=1:1:6503
        if Charge_komplett(j,1)==1
        y=y+1;
        Einleger(x,y)=j;
         if y==4
         y=0;
         x=x+1;
        end
        end
    end

filename='Taguchi30-4.gex';
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
for i=197:1:245 %n^2
   
    positionx_aktuell=ParameterArray(i,1);
    positiony_aktuell=ParameterArray(i,2);
    designvariablen_aktuell=ParameterArray(i,3:10);
    [Charge,Charge_komplett]=variabletransformation(positionx_aktuell, positiony_aktuell, designvariablen_aktuell, Elemente, Elementenanzahl, Knoten, Loeschliste,i);
    Area=sum(Charge_komplett);
    Einleger=zeros(1,1);
    Hight(i)=(6503/Area)*3;
    x=1;
    y=0;
    for j=1:1:6503
        if Charge_komplett(j,1)==1
        y=y+1;
        Einleger(x,y)=j;
         if y==4
         y=0;
         x=x+1;
        end
        end
    end

filename='Taguchi30-5.gex';
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
for i=246:1:294 %n^2
   
    positionx_aktuell=ParameterArray(i,1);
    positiony_aktuell=ParameterArray(i,2);
    designvariablen_aktuell=ParameterArray(i,3:10);
    [Charge,Charge_komplett]=variabletransformation(positionx_aktuell, positiony_aktuell, designvariablen_aktuell, Elemente, Elementenanzahl, Knoten, Loeschliste,i);
    Area=sum(Charge_komplett);
    Einleger=zeros(1,1);
    Hight(i)=(6503/Area)*3;
    x=1;
    y=0;
    for j=1:1:6503
        if Charge_komplett(j,1)==1
        y=y+1;
        Einleger(x,y)=j;
         if y==4
         y=0;
         x=x+1;
        end
        end
    end

filename='Taguchi30-6.gex';
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
for i=295:1:343 %n^2
   
    positionx_aktuell=ParameterArray(i,1);
    positiony_aktuell=ParameterArray(i,2);
    designvariablen_aktuell=ParameterArray(i,3:10);
    [Charge,Charge_komplett]=variabletransformation(positionx_aktuell, positiony_aktuell, designvariablen_aktuell, Elemente, Elementenanzahl, Knoten, Loeschliste,i);
    Area=sum(Charge_komplett);
    Einleger=zeros(1,1);
    Hight(i)=(6503/Area)*3;
    x=1;
    y=0;
    for j=1:1:6503
        if Charge_komplett(j,1)==1
        y=y+1;
        Einleger(x,y)=j;
         if y==4
         y=0;
         x=x+1;
        end
        end
    end

filename='Taguchi30-7.gex';
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
for i=344:1:392 %n^2
   
    positionx_aktuell=ParameterArray(i,1);
    positiony_aktuell=ParameterArray(i,2);
    designvariablen_aktuell=ParameterArray(i,3:10);
    [Charge,Charge_komplett]=variabletransformation(positionx_aktuell, positiony_aktuell, designvariablen_aktuell, Elemente, Elementenanzahl, Knoten, Loeschliste,i);
    Area=sum(Charge_komplett);
    Einleger=zeros(1,1);
    Hight(i)=(6503/Area)*3;
    x=1;
    y=0;
    for j=1:1:6503
        if Charge_komplett(j,1)==1
        y=y+1;
        Einleger(x,y)=j;
         if y==4
         y=0;
         x=x+1;
        end
        end
    end

filename='Taguchi30-8.gex';
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
for i=393:1:441 %n^2
   
    positionx_aktuell=ParameterArray(i,1);
    positiony_aktuell=ParameterArray(i,2);
    designvariablen_aktuell=ParameterArray(i,3:10);
    [Charge,Charge_komplett]=variabletransformation(positionx_aktuell, positiony_aktuell, designvariablen_aktuell, Elemente, Elementenanzahl, Knoten, Loeschliste,i);
    Area=sum(Charge_komplett);
    Einleger=zeros(1,1);
    Hight(i)=(6503/Area)*3;
    x=1;
    y=0;
    for j=1:1:6503
        if Charge_komplett(j,1)==1
        y=y+1;
        Einleger(x,y)=j;
         if y==4
         y=0;
         x=x+1;
        end
        end
    end

filename='Taguchi30-9.gex';
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
for i=442:1:490 %n^2
   
    positionx_aktuell=ParameterArray(i,1);
    positiony_aktuell=ParameterArray(i,2);
    designvariablen_aktuell=ParameterArray(i,3:10);
    [Charge,Charge_komplett]=variabletransformation(positionx_aktuell, positiony_aktuell, designvariablen_aktuell, Elemente, Elementenanzahl, Knoten, Loeschliste,i);
    Area=sum(Charge_komplett);
    Einleger=zeros(1,1);
    Hight(i)=(6503/Area)*3;
    x=1;
    y=0;
    for j=1:1:6503
        if Charge_komplett(j,1)==1
        y=y+1;
        Einleger(x,y)=j;
         if y==4
         y=0;
         x=x+1;
        end
        end
    end

filename='Taguchi30-10.gex';
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
for i=491:1:539 %n^2
   
    positionx_aktuell=ParameterArray(i,1);
    positiony_aktuell=ParameterArray(i,2);
    designvariablen_aktuell=ParameterArray(i,3:10);
    [Charge,Charge_komplett]=variabletransformation(positionx_aktuell, positiony_aktuell, designvariablen_aktuell, Elemente, Elementenanzahl, Knoten, Loeschliste,i);
    Area=sum(Charge_komplett);
    Einleger=zeros(1,1);
    Hight(i)=(6503/Area)*3;
    x=1;
    y=0;
    for j=1:1:6503
        if Charge_komplett(j,1)==1
        y=y+1;
        Einleger(x,y)=j;
         if y==4
         y=0;
         x=x+1;
        end
        end
    end

filename='Taguchi30-11.gex';
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
for i=540:1:588 %n^2
   
    positionx_aktuell=ParameterArray(i,1);
    positiony_aktuell=ParameterArray(i,2);
    designvariablen_aktuell=ParameterArray(i,3:10);
    [Charge,Charge_komplett]=variabletransformation(positionx_aktuell, positiony_aktuell, designvariablen_aktuell, Elemente, Elementenanzahl, Knoten, Loeschliste,i);
    Area=sum(Charge_komplett);
    Einleger=zeros(1,1);
    Hight(i)=(6503/Area)*3;
    x=1;
    y=0;
    for j=1:1:6503
        if Charge_komplett(j,1)==1
        y=y+1;
        Einleger(x,y)=j;
         if y==4
         y=0;
         x=x+1;
        end
        end
    end

filename='Taguchi30-12.gex';
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
for i=589:1:637 %n^2
   
    positionx_aktuell=ParameterArray(i,1);
    positiony_aktuell=ParameterArray(i,2);
    designvariablen_aktuell=ParameterArray(i,3:10);
    [Charge,Charge_komplett]=variabletransformation(positionx_aktuell, positiony_aktuell, designvariablen_aktuell, Elemente, Elementenanzahl, Knoten, Loeschliste,i);
    Area=sum(Charge_komplett);
    Einleger=zeros(1,1);
    Hight(i)=(6503/Area)*3;
    x=1;
    y=0;
    for j=1:1:6503
        if Charge_komplett(j,1)==1
        y=y+1;
        Einleger(x,y)=j;
         if y==4
         y=0;
         x=x+1;
        end
        end
    end

filename='Taguchi30-13.gex';
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
for i=638:1:686 %n^2
   
    positionx_aktuell=ParameterArray(i,1);
    positiony_aktuell=ParameterArray(i,2);
    designvariablen_aktuell=ParameterArray(i,3:10);
    [Charge,Charge_komplett]=variabletransformation(positionx_aktuell, positiony_aktuell, designvariablen_aktuell, Elemente, Elementenanzahl, Knoten, Loeschliste,i);
    Area=sum(Charge_komplett);
    Einleger=zeros(1,1);
    Hight(i)=(6503/Area)*3;
    x=1;
    y=0;
    for j=1:1:6503
        if Charge_komplett(j,1)==1
        y=y+1;
        Einleger(x,y)=j;
         if y==4
         y=0;
         x=x+1;
        end
        end
    end

filename='Taguchi30-14.gex';
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
for i=687:1:735 %n^2
   
    positionx_aktuell=ParameterArray(i,1);
    positiony_aktuell=ParameterArray(i,2);
    designvariablen_aktuell=ParameterArray(i,3:10);
    [Charge,Charge_komplett]=variabletransformation(positionx_aktuell, positiony_aktuell, designvariablen_aktuell, Elemente, Elementenanzahl, Knoten, Loeschliste,i);
    Area=sum(Charge_komplett);
    Einleger=zeros(1,1);
    Hight(i)=(6503/Area)*3;
    x=1;
    y=0;
    for j=1:1:6503
        if Charge_komplett(j,1)==1
        y=y+1;
        Einleger(x,y)=j;
         if y==4
         y=0;
         x=x+1;
        end
        end
    end

filename='Taguchi30-15.gex';
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
for i=736:1:784 %n^2
   
    positionx_aktuell=ParameterArray(i,1);
    positiony_aktuell=ParameterArray(i,2);
    designvariablen_aktuell=ParameterArray(i,3:10);
    [Charge,Charge_komplett]=variabletransformation(positionx_aktuell, positiony_aktuell, designvariablen_aktuell, Elemente, Elementenanzahl, Knoten, Loeschliste,i);
    Area=sum(Charge_komplett);
    Einleger=zeros(1,1);
    Hight(i)=(6503/Area)*3;
    x=1;
    y=0;
    for j=1:1:6503
        if Charge_komplett(j,1)==1
        y=y+1;
        Einleger(x,y)=j;
         if y==4
         y=0;
         x=x+1;
        end
        end
    end

filename='Taguchi30-16.gex';
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
for i=785:1:833 %n^2
   
    positionx_aktuell=ParameterArray(i,1);
    positiony_aktuell=ParameterArray(i,2);
    designvariablen_aktuell=ParameterArray(i,3:10);
    [Charge,Charge_komplett]=variabletransformation(positionx_aktuell, positiony_aktuell, designvariablen_aktuell, Elemente, Elementenanzahl, Knoten, Loeschliste,i);
    Area=sum(Charge_komplett);
    Einleger=zeros(1,1);
    Hight(i)=(6503/Area)*3;
    x=1;
    y=0;
    for j=1:1:6503
        if Charge_komplett(j,1)==1
        y=y+1;
        Einleger(x,y)=j;
         if y==4
         y=0;
         x=x+1;
        end
        end
    end

filename='Taguchi30-17.gex';
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
for i=834:1:882 %n^2
   
    positionx_aktuell=ParameterArray(i,1);
    positiony_aktuell=ParameterArray(i,2);
    designvariablen_aktuell=ParameterArray(i,3:10);
    [Charge,Charge_komplett]=variabletransformation(positionx_aktuell, positiony_aktuell, designvariablen_aktuell, Elemente, Elementenanzahl, Knoten, Loeschliste,i);
    Area=sum(Charge_komplett);
    Einleger=zeros(1,1);
    Hight(i)=(6503/Area)*3;
    x=1;
    y=0;
    for j=1:1:6503
        if Charge_komplett(j,1)==1
        y=y+1;
        Einleger(x,y)=j;
         if y==4
         y=0;
         x=x+1;
        end
        end
    end

filename='Taguchi30-18.gex';
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
for i=883:1:900 %n^2
   
    positionx_aktuell=ParameterArray(i,1);
    positiony_aktuell=ParameterArray(i,2);
    designvariablen_aktuell=ParameterArray(i,3:10);
    [Charge,Charge_komplett]=variabletransformation(positionx_aktuell, positiony_aktuell, designvariablen_aktuell, Elemente, Elementenanzahl, Knoten, Loeschliste,i);
    Area=sum(Charge_komplett);
    Einleger=zeros(1,1);
    Hight(i)=(6503/Area)*3;
    x=1;
    y=0;
    for j=1:1:6503
        if Charge_komplett(j,1)==1
        y=y+1;
        Einleger(x,y)=j;
         if y==4
         y=0;
         x=x+1;
        end
        end
    end

filename='Taguchi30-19.gex';
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

save('MetadatenTaguchi30.mat','ParameterArray', 'Hight');
save('MetadatenTaguchi30.xslx','ParameterArray', 'Hight');