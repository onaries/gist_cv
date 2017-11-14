
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd


# In[2]:


cv2.__version__


# In[3]:


img1 = cv2.imread('img1.png', cv2.IMREAD_COLOR)
img2 = cv2.imread('img2.png', cv2.IMREAD_COLOR)


# In[4]:


# opencv : BGR, matplotlib : RGB
b, g, r = cv2.split(img1)
img1 = cv2.merge([r, g, b])
b, g, r = cv2.split(img2)
img2 = cv2.merge([r, g, b])


# In[5]:


gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# In[6]:


img1_file = plt.imread('img1.png')
img2_file = plt.imread('img2.png')


# In[7]:


plt.figure(figsize=(15,15))
plt.subplot(121)
plt.imshow(img1)
plt.title('img1')

plt.subplot(122)
plt.imshow(img2)
plt.title('img2')
plt.show()


# In[8]:


sift = cv2.xfeatures2d.SIFT_create()
(kp1, descs1) = sift.detectAndCompute(gray1, None)
sift_img = cv2.drawKeypoints(img1, kp1, None)

plt.figure(figsize=(10, 10))
plt.imshow(sift_img)
plt.title('keypoints of img1')
plt.show()


# In[9]:


sift2 = cv2.xfeatures2d.SIFT_create()
(kp2, descs2) = sift.detectAndCompute(gray2, None)
sift_img2 = cv2.drawKeypoints(img2, kp2, None)

plt.figure(figsize=(10, 10))
plt.imshow(sift_img2)
plt.title('keypoints of img2')
plt.show()


# In[10]:


bf = cv2.BFMatcher()
matches = bf.knnMatch(descs1, descs2, k=2)


# In[11]:


good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])


# In[12]:


img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good, None)
plt.figure(figsize=(15,15))
plt.imshow(img3)
plt.title('matches')
plt.show()


# In[21]:


# 매칭점 인덱스 알아내기
pt = []

for i in range(10, 18):
    pt.append((good[i][0].queryIdx, good[i][0].trainIdx))


# In[22]:


# 매칭점 좌표 알아내기
coordi = []

for i in range(8):
    coordi.append((np.round(kp1[pt[i][0]].pt), np.round(kp2[pt[i][1]].pt)))


# In[23]:


coordi


# In[24]:


# 매징점 좌표 표시
plt.figure(figsize=(15,15))
plt.subplot(121)
plt.imshow(img1)
for i in range(8):
    plt.scatter(x=coordi[i][0][0], y=coordi[i][0][1])

plt.subplot(122)
plt.imshow(img2)
for i in range(8):
    plt.scatter(x=coordi[i][1][0], y=coordi[i][1][1])
plt.show()


# In[25]:


# A Matrix 구하기
A = []

for i in range(8):
    A.append([coordi[i][0][0]*coordi[i][1][0], coordi[i][0][0]*coordi[i][1][1], coordi[i][0][0], coordi[i][0][1]*coordi[i][1][0], coordi[i][0][1]*coordi[i][1][1], coordi[i][0][1], coordi[i][1][0], coordi[i][1][1], 1])


# In[26]:


# A 
df = pd.DataFrame(data=A)
df.style


# In[27]:


# SVD(A)
U, S, V = np.linalg.svd(A)


# In[28]:


# 

F = V.T[:, -1]
F = np.array(F)
print(F)


# In[29]:


F = F.reshape(3, 3)
df = pd.DataFrame(data=F)
df.style


# In[30]:


# line
line = []
for i in range(8):
    line.append(np.dot(F, np.array([[coordi[i][1][0]], [coordi[i][1][1]], [1]])))


# In[31]:


line= np.array(line)
line = line.reshape(8, 3)


# In[32]:


# line vector

df = pd.DataFrame(data=line)
df


# In[33]:


plt.figure(figsize=(10, 10))
plt.imshow(img1)
for i in range(8):
    plt.scatter(x=coordi[i][0][0], y=coordi[i][0][1])
    for j in range(1, 640):
        y = -(line[i][0]*j + line[i][2]) / line[i][1]
        plt.plot([j], [y], marker='1', markersize=1, color='blue')
        
plt.show()


# In[34]:


np_img1 = np.zeros((3, 8))
np_img2 = np.zeros((3, 8))

for i in range(8):
    np_img1[:, i] = [coordi[i][0][0], coordi[i][0][1], 1]
    np_img2[:, i] = [coordi[i][1][0], coordi[i][1][1], 1]
    
np_img1 = np.array(np_img1)
np_img2 = np.array(np_img2)


# In[35]:


# Normalized 

def normalize(pts):
    
    # centroid 구하기
    C = [np.mean(pts[0]), np.mean(pts[1])]
    print(C)
    
    new_pts = np.zeros((3, 8))
    
    # shift the origin to centroid
    new_pts[0] = pts[0] - C[0]
    new_pts[1] = pts[1] - C[1]
    
    mean_dist = np.mean(np.sqrt(np.power(new_pts[0], 2) + np.power(new_pts[1], 2)))
    print(mean_dist)
    scale = np.sqrt(2) / mean_dist
    print(scale)
    
    T = [[scale, 0, -scale*C[0]], [0, scale, -scale*C[1]], [0, 0, 1]]
    print(T)
    npts = np.dot(T, pts)
    
    return npts, T


# In[36]:


npt_left, T1 = normalize(np_img1)


# In[37]:


npt_right, T2 = normalize(np_img2)


# In[38]:


nA = np.zeros((8, 9))

for i in range(8):
    nx_left = npt_left[0, i]
    ny_left = npt_left[1, i]
    nx_right = npt_right[0, i]
    ny_right = npt_right[1, i]
    
    nA[i, :] = [nx_left*nx_right, nx_left*ny_right, nx_left, ny_left*nx_right, ny_left*ny_right, ny_left, nx_right, ny_right, 1]


# In[39]:


Un, Dn, Vn = np.linalg.svd(nA)


# In[40]:


nF = Vn.conj().T[:, -1].reshape(3, 3)


# In[41]:


nF


# In[45]:


nU, nD, nV = np.linalg.svd(nF)


# In[46]:


nF = np.dot(np.dot(nU, np.diag([nD[0], nD[1], 0])),nV.conj().T)


# In[42]:


nF = np.dot(np.dot(np.array(T2).conj().T, nF), np.array(T1))


# In[43]:


nF


# In[44]:


df = pd.DataFrame(data=F)
df


# In[45]:


df = pd.DataFrame(data=nF)
df


# In[46]:


nLine = []

for i in range(8):
    nLine.append(np.dot(F.T, np.array([[coordi[i][0][0]], [coordi[i][0][1]], [1]])))


# In[47]:


nLine= np.array(nLine)
nLine.reshape(8, 3)


# In[48]:


plt.figure(figsize=(10, 10))
plt.imshow(img2)

for i in range(8):
    plt.scatter(x=coordi[i][1][0], y=coordi[i][1][1])
    for j in range(1, 640):
        y = - (nLine[i][0]*j + nLine[i][2]) / nLine[i][1]
        plt.plot([j], [y], marker='1', markersize=1, color='red')

plt.show()


# In[49]:


nLine = []

for i in range(8):
    nLine.append(np.dot(nF, np.array([[coordi[i][0][0]], [coordi[i][0][1]], [1]])))

nLine= np.array(nLine)
nLine.reshape(8, 3)

plt.figure(figsize=(10, 10))
plt.imshow(img1)

for i in range(8):
    plt.scatter(x=coordi[i][0][0], y=coordi[i][0][1])
    for j in range(1, 640):
        y = - (nLine[i][0]*j + nLine[i][2]) / nLine[i][1]
        plt.plot([j], [y], marker='1', markersize=1, color='red')

plt.show()

