clear all;
clc;

% pixel coordinate
points = [453 353; 781 190; 1136 368; 760 532; 467 714; 786 945; 1068 719];

% world coordinate
world_points = [48 0 48; 0 0 48; 0 48 48; 0 0 0; 48 0 0; 48 48 0; 0 48 0];

% Initialize Matrix
A = zeros(2*7, 12);
P = zeros(3, 4);

% Calculate A Matrix
for i=1:7
  x = -points(i, 1);
  y = -points(i, 2);
  world_x = world_points(i, 1);
  world_y = world_points(i, 2);
  world_z = world_points(i, 3);

  A(2*i-1, :) = [world_x world_y world_z 1 0 0 0 0 world_x*x world_y*x world_z*x x];
  A(2*i, :) = [0 0 0 0 world_x world_y world_z 1 world_x*y world_y*y world_z*y y];
end

% SVD(A)
[U1, W1, V1] = svd(A);

% Find out P Matrix
for i=1:4
  P(1, i) = V1(i, 12);
  P(2, i) = V1(i+4, 12);
  P(3, i) = V1(i+8, 12);
end

% SVD(P)
[U2, W2, V2] = svd(P);

% Camera center C
C = [V2(1, 4); V2(2, 4); V2(3, 4); V2(4, 4)];
C = C/C(4); % homogeneous coordinate
 
tC = [C(1); C(2); C(3)]; % tilt C

i3_tC = [1 0 0 -tC(1); 0 1 0 -tC(2); 0 0 1 -tC(3)];

KR = P*pinv(i3_tC);

[K, R] = rq(KR);
K = K/K(3,3);
t = -R*tC;

ground_truth = [166.20; 141.46; 170.08; 1];
err = abs(ground_truth - C);

% print
format bank

P
K
R
t
C
ground_truth
err