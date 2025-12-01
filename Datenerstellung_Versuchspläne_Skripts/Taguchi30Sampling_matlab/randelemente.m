function [output] = randelemente(Elemente, Elementenanzahl, Knoten,u1,u2,output)
i=0;
for q=1:1:Elementenanzahl
    for k=1:1:2
        h1=[Knoten(Elemente(q,k+1),2);Knoten(Elemente(q,k+1),3)];
        h2=[Knoten(Elemente(q,k+2),2)-Knoten(Elemente(q,k+1),2);Knoten(Elemente(q,k+2),3)-Knoten(Elemente(q,k+1),3)];
        if h2(2)==0
        n=(h1(2)+((h2(2)*u1(1))/h2(1))-u1(2)-(h2(2)*h1(1))/h2(1))/(u2(2)-((h2(2)*u2(1))/h2(1)));
        else
        n=(h1(1)+((h2(1)*u1(2))/h2(2))-u1(1)-(h2(1)*h1(2))/h2(2))/(u2(1)-((h2(1)*u2(2))/h2(2)));
        end

        if h2(1)==0
        m=(u1(2)-h1(2)+n*u2(2))/h2(2);
        else
        m=(u1(1)-h1(1)+n*u2(1))/h2(1);
        end
        n=round(n*10000000)/10000000;
        m=round(m*10000000)/10000000;
        if isempty(m)
            m=0;
            n=0;
        end
        if n>0 && n<1 && m>0 && m<1
            i=i+1;
            output(q,1) = 1;
        end
    end
    
        h1=[Knoten(Elemente(q,4),2);Knoten(Elemente(q,4),3)];
        h2=[Knoten(Elemente(q,2),2)-Knoten(Elemente(q,4),2);Knoten(Elemente(q,2),3)-Knoten(Elemente(q,4),3)];
        
        if h2(2)==0
        n=(h1(2)+((h2(2)*u1(1))/h2(1))-u1(2)-(h2(2)*h1(1))/h2(1))/(u2(2)-((h2(2)*u2(1))/h2(1)));
        else
        n=(h1(1)+((h2(1)*u1(2))/h2(2))-u1(1)-(h2(1)*h1(2))/h2(2))/(u2(1)-((h2(1)*u2(2))/h2(2)));
        end

        if h2(1)==0
        m=(u1(2)-h1(2)+n*u2(2))/h2(2);
        else
        m=(u1(1)-h1(1)+n*u2(1))/h2(1);
        end
        n=round(n*10000000)/10000000;
        m=round(m*10000000)/10000000;
        if isempty(m)
            m=0;
            n=0;
        end
        if n>0 && n<1 && m>0 && m<1
            output(q,1) = 1;
        end
    
end