function [output_neu,output_alt] = variabletransformation_mit_Winkel(positionx, positiony, designvariablen, Elemente, Elementenanzahl, Knoten,Loeschliste, Nummer, alpha)
% Übertragung der Designvariablendefinition auf die Variablendefinition des NN
output_alt = zeros(Elementenanzahl,1);
%% Positionsvektor
u=[positionx; positiony];
%% Formvektoren
numVector = size(designvariablen,2);
v=zeros(2,numVector);
for i=1:numVector
    v1 = cos(((i-1)*2*pi/numVector)+(alpha/180)*pi)*designvariablen(1,i);
    v2 = sin(((i-1)*2*pi/numVector)+(alpha/180)*pi)*designvariablen(1,i);
    v(:,i)= [v1 ; v2];
end
%% Identifikation der inneren Elemente
for a=1:numVector-1
    for i=1:Elementenanzahl
        A = [v(1,a) v(1,a+1) ; v(2,a) v(2,a+1)];
        b = [Elemente(i,7)-u(1) ; Elemente(i,8)-u(2)];
        x=A\b;
        %x(1)=((b(1)/A(1,1))-((A(1,2)*b(2)))/(A(2,2)*A(1,1)))/(1-(A(1,2)*A(2,1))/(A(2,2)*A(1,1)));
        %x(2)=(b(2)-A(2,1)*x(1))/(A(2,2));
        if (x(1)>=0) && (x(1)<=1) && (x(2)>=0)  && x(1)<= 1-x(2)
            output_alt(i,1) = 1;
        end
    end
end
a=numVector;
for i=1:Elementenanzahl
    A = [v(1,a) v(1,1) ; v(2,a) v(2,1)];
    b = [Elemente(i,7)-u(1) ; Elemente(i,8)-u(2)];
    x=A\b;
%     x(1)=((b(1)/A(1,1))-((A(1,2)*b(2)))/(A(2,2)*A(1,1)))/(1-(A(1,2)*A(2,1))/(A(2,2)*A(1,1)));
%     x(2)=(b(2)-A(2,1)*x(1))/(A(2,2));
    if (x(1)>=0) && (x(1)<=1) && (x(2)>=0) && x(1)<= 1-x(2)
        output_alt(i,1) = 1;
    end
end
%% Identifikation der Randelemete
for p=1:numVector-1
    umrandung1=u+v(:,p);
    umrandung2=(v(:,p+1)-v(:,p));
    output_alt=randelemente(Elemente, Elementenanzahl, Knoten,umrandung1,umrandung2,output_alt);
end
umrandung1=u+v(:,numVector);
umrandung2=(v(:,1)-v(:,numVector));
output_alt=randelemente(Elemente, Elementenanzahl, Knoten,umrandung1,umrandung2,output_alt);
% %% Ausschluss von nicht in der Ebene liegenden Elementen
% for i=1:1:Elementenanzahl
%     if Elemente(i,9)>1
%         output_alt(i,1)=0;  
%     end
% end
%% Ausschluss von unbelegbaren Elementen und einfügen Abstand

output_neu=zeros(Elementenanzahl,1);
for i=1:1:Elementenanzahl
output_neu(i,1)=output_alt(i,1)*Elemente(i,9);
end
x=0;
for i=1:1:length(Loeschliste)
    output_neu(Loeschliste(1,i)-x,:)=[];
    output_alt(Loeschliste(1,i),:)=0;
    x=x+1;
end
% %% Visualisierung
   %visualisierungCharge(output_alt, Knoten, Elemente, Elementenanzahl,Nummer);
   %visualisierunggrid(output_alt,Knoten, Elemente, Elementenanzahl,u,v,numVector);
   visualisierung(output_alt, Knoten, Elemente, Elementenanzahl,u,v, numVector, Nummer)
