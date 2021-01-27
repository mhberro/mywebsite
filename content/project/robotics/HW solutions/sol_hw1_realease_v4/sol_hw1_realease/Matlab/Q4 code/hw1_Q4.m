% this script is for 6740 2020summer, HW1 Q4

clear; close all;
T = readtable('food-consumption.csv','ReadRowNames',...
    true,'ReadVariableNames',true, 'PreserveVariableNames', true);
T = rmmissing(T);
data = table2array(T);                      
countrynames = T.Properties.RowNames;        
foodnames = T.Properties.VariableNames;
Anew= data; 
[m,n]=size(Anew);
stdA = std(Anew, 1, 1); 
Anew = Anew * diag(1./stdA); 
Anew = Anew'; 
mu=sum(Anew,2)./m;
xc = bsxfun(@minus, Anew, mu); 
                 
C = xc * xc' ./ m; 
k = 2; 
[W, S] = eigs(C, k); 
S = diag(S,0);

dim1 = W(:,1)' * xc ./ sqrt(S(1));
dim2 = W(:,2)' * xc ./ sqrt(S(2));


%% Q4b
figure();
subplot(2,1,1)
stem(W(:,1),'b*')
set(gca,'xtick',1:20,'xticklabel',foodnames)
xtickangle(45)
title('1st pc')
axis square

subplot(2,1,2)
stem(W(:,2),'b*')
set(gca,'xtick',1:20,'xticklabel',foodnames)
xtickangle(45)
title('2nd pc')
axis square

%% Q4c
figure();
plot(dim1,dim2,'r*', 'MarkerSize', 5); 
text(dim1 ,dim2,countrynames,'FontSize',10,'VerticalAlignment','top');  
xlabel('1st pc', 'fontsize', 14); ylabel('2nd pc', 'fontsize', 14)


