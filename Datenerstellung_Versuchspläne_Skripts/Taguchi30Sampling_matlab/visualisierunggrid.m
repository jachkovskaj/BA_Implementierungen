function [] = visualisierunggrid(Charge, Knoten, Elemente, Elementenanzahl,u,v, numVector)


%% Aufrufen des Plots
%output=v;
fg=figure(3);
set(fg, 'Position', [401, 451, 1445, 453]);
%figure('Name','Visualisierung','NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
clf
hold on
%% Elemente plot
% for i=1:Elementenanzahl
%     x = [Knoten(Elemente(i,2),2) Knoten(Elemente(i,3),2) Knoten(Elemente(i,4),2) ];
%     y = [Knoten(Elemente(i,2),3) Knoten(Elemente(i,3),3) Knoten(Elemente(i,4),3) ];
%     patch(x,y,'w')
%     
% end

%% Mittelpunkt der Elemente plot
%plot(Elemente(:,7),Elemente(:,8),'b.')

%%  Markierung der ausgewählten Elemente
% for i=1:Elementenanzahl
%     if Charge(i,1)==1
%         x = [Knoten(Elemente(i,2),2) Knoten(Elemente(i,3),2) Knoten(Elemente(i,4),2) ];
%         y = [Knoten(Elemente(i,2),3) Knoten(Elemente(i,3),3) Knoten(Elemente(i,4),3) ];
%         patch(x,y,'g')
%         %plot(Elemente(i,7),Elemente(i,8),'r.')
%     end
% end

%% Vektoren plot
plot([0;u(1)],[0;u(2)],'k.-','LineWidth',1.5)

for i=1:numVector
    
    plot([u(1);u(1)+v(1,i)],[u(2);u(2)+v(2,i)],'k.-','LineWidth',1.5)
    
end

%% Umrandung plot

for i=1:numVector-1
    
    plot([u(1)+v(1,i);u(1)+v(1,i+1)],[u(2)+v(2,i);u(2)+v(2,i+1)],'k.-','LineWidth',2)
end
plot([u(1)+v(1,numVector);u(1)+v(1,1)],[u(2)+v(2,numVector);u(2)+v(2,1)],'k.-','LineWidth',2)
 %% Plot Skalierung
 %axis([-350 350 -150 150])
  %% Plot Skalierung
 axis([-1050 1050 -450 450])