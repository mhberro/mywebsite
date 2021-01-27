function [ class, centroid ] = mykmedoids( pixels, K)
% % this script is for 6740 2020summer, HW1 Q2


c1=datasample(pixels,K);
c0=c1-10;

z=1;
while (norm(c1 - c0, 'fro') > 1e-6)
%     fprintf(1, '--iteration %d \n', z);
    
    % record previous c; 
    c0 = c1; 
    
    % assign data points to current cluster; 
    for j = 1:length(pixels) % loop through data points; 
        tmp_distance = zeros(1, K); 
        for k = 1:K % through centers; 
%             tmp_distance(k) = sum((pixels(j,:) - c1(k,:)).^2); % norm(x(:,j) - c(:,k)); 
            tmp_distance(k) = max(abs(pixels(j,:) - c1(k,:))); % inf distance 
        end
        [~,K_index] = min(tmp_distance); % ~ ignores the first argument; 
        P(:, j) = zeros(K, 1); 
        P(K_index, j) = 1; 
    end
        
    % adjust the cluster centers according to current assignment; 

    obj=0;
    obj2=0;
    for k = 1:K
        idx = find(P(k, :)>0); 
        no_of_points = length(idx);             
        centroid(k,:) = P(k,:) * pixels ./ no_of_points; 
        tmp_distance2 = zeros(1, K); 
        for l=1:length(idx)
%             tmp_distance2(l) = sum((pixels(idx(l),:) - centroid(k,:)).^2);
            tmp_distance2(l) = max(abs(pixels(idx(l),:) - centroid(k,:)));
        end
        [~,cntr_pt_index] = min(tmp_distance2); % ~ ignores the first argument;
        cnew(k,:)=pixels(idx(cntr_pt_index),:);
%         obj = obj + sum(sum((pixels(idx,:) - repmat(c1(k,:),no_of_points,1)).^2));
%         obj2 = obj2 + sum(sum((pixels(idx,:) - repmat(cnew(k,:),no_of_points,1)).^2));
        obj = obj + sum(max(abs(pixels(idx,:) - repmat(c1(k,:),no_of_points,1))));
        obj2 = obj2 + sum(max(abs(pixels(idx,:) - repmat(cnew(k,:),no_of_points,1))));
        clear tmp_distance2;
    end

    
    if obj2-obj<0
        c1=cnew;
    end
    
    z = z + 1;     
end   
fprintf(1, '\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~');
fprintf(1, 'Kmedoids, K= %d \n', K);
fprintf(1, '# of iterations %d \n', z);
P1=(sum(P.*(1:K)'))';
    
    class=P1;
    centroid=c1;
end