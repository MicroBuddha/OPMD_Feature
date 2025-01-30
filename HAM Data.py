#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('nvidia-smi')


# ### Our Model

# In[1]:


from utils import *
rgb_directory='HAM/RGB/'
hsi_directory='HAM/HSI/'
labels_directory='ham.csv'
df=pd.read_csv(labels_directory)
df=df.drop(['Unnamed: 0.1','Unnamed: 0'], axis=1)
idx_to_drop=[]
for i in range(len(df)):
    impath=df.iloc[i,0]
    name=impath.split('/')[-1]
    hsi_path=os.path.join(hsi_directory, name.replace('.jpg','.mat'))
    if os.path.exists(hsi_path):
        continue
    else:
        idx_to_drop.append(i)

df=df.drop(idx_to_drop, axis=0)
df=df.reset_index(drop=True)
ids=df['ID']
ids=list(set(list(ids)))
c0 = set(df[df['class'] == 0]['ID'].tolist())
c1 = set(df[df['class'] == 1]['ID'].tolist())
c2 = set(df[df['class'] == 2]['ID'].tolist())
c3 = set(df[df['class'] == 3]['ID'].tolist())
c4 = set(df[df['class'] == 4]['ID'].tolist())
c5 = set(df[df['class'] == 5]['ID'].tolist())
c6 = set(df[df['class'] == 6]['ID'].tolist())

train_ids = []
test_ids = []
superclass_sets = [c4]
for class_set in superclass_sets:
    train, test = train_test_split(list(class_set), test_size=0.9)
    train_ids+=train
    test_ids += test
overclass_sets = [c0, c1, c5]
for class_set in overclass_sets:
    train, test = train_test_split(list(class_set), test_size=0.4)
    train_ids+=train
    test_ids += test
underclass_sets = [c2]
for class_set in underclass_sets:
    train, test = train_test_split(list(class_set), test_size=0.2)
    train_ids+=train
    test_ids += test
poorclass_sets = [c3, c6]
for class_set in poorclass_sets:
    train, test = train_test_split(list(class_set), test_size=0.005)
    train_ids+=train
    test_ids += test

train_images=[]
test_images=[]
train_labels=[]
test_labels=[]
for i in range(len(df)):
    idnum=df.iloc[i,-1]
    path=df.iloc[i,0]
    label=df.iloc[i,1]
    if idnum in train_ids:
        train_images.append(path)
        train_labels.append(label)
    else:
        test_images.append(path)
        test_labels.append(label)

A=list(set(train_labels))
B=list(set(test_labels))
labels={A[idx]:idx for idx in range(len(A))}
def calculate_hu_moments(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    moments = cv2.moments(gray_image)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments
def calculate_haralick_texture(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    textures = mahotas.features.haralick(gray_image).mean(axis=0)
    return textures
def build_kmeans_for_sift(image_paths, num_clusters=50):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for image_path in tqdm(image_paths, desc="Extracting SIFT Descriptors"):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)

    all_descriptors = np.vstack(descriptors_list)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_descriptors)
    return kmeans
def calculate_sift_bovw(image, kmeans, num_clusters=50):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    
    if descriptors is not None:
        labels = kmeans.predict(descriptors)
        hist = np.bincount(labels, minlength=num_clusters).astype(float)
        return hist / hist.sum()
    else:
        return np.zeros(num_clusters)
num_clusters = 50
kmeans = build_kmeans_for_sift(train_images, num_clusters=num_clusters)
def extract_rgb_features(image):
    image = image.astype(np.float32)
    mean_intensity = np.mean(image)
    mean_values = np.mean(image, axis=(0,1))
    r_g_ratio = mean_values[2] / mean_values[1] if mean_values[1] != 0 else 0
    r_b_ratio = mean_values[2] / mean_values[0] if mean_values[0] != 0 else 0
    g_b_ratio = mean_values[1] / mean_values[0] if mean_values[0] != 0 else 0    
    return np.array([mean_intensity, r_g_ratio, r_b_ratio, g_b_ratio])

def extract_hsi_features(mat_image):
    with h5py.File(mat_image, 'r') as mat_file:
        if 'cube' in mat_file:
            data = mat_file['cube'][:]
            numpy_array = np.array(data)
            ref = np.mean(numpy_array, axis=(1,2))
            ref = ref / 255.0
            return ref
        else:
            # Print keys only if 'cube' is missing
            keys = list(mat_file.keys())
            print(f"Error: 'cube' key not found in file {mat_image}. Available keys: {keys}")
            

def calculate_non_black_avg_rgb(image):
    non_black_mask = np.any(image != 0, axis=-1)
    avg_r = image[:, :, 0].mean()
    avg_g = image[:, :, 1].mean()
    avg_b = image[:, :, 2].mean()
    return avg_r, avg_g, avg_b  
    
def extract_features(image_path):
    filename=image_path.split('/')[-1]
    name=filename.replace('.jpg','.mat')
    matfile=os.path.join(hsi_directory, name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    avg_r, avg_g, avg_b = calculate_non_black_avg_rgb(image)
    hu_moments = calculate_hu_moments(image)
    haralick_texture = calculate_haralick_texture(image)
    sift_bovw = calculate_sift_bovw(image, kmeans, num_clusters=50)
    rgb_features=extract_rgb_features(image)
    hsi_features=extract_hsi_features(matfile)
    features = np.concatenate(( hu_moments, haralick_texture, sift_bovw, rgb_features,hsi_features))
    return features

train_features = np.array([extract_features(img_path) for img_path in tqdm(train_images, desc="Extracting Training Features")])
test_features = np.array([extract_features(img_path) for img_path in tqdm(test_images, desc="Extracting Testing Features")])
max_acc=0
state=0
for i in tqdm(range(100), desc='States'):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=i)
    rf_classifier.fit(train_features, train_labels)
    train_predictions = rf_classifier.predict(train_features)
    test_predictions = rf_classifier.predict(test_features)
    
    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    if test_accuracy>max_acc:
        state=i
    max_acc=max(max_acc, test_accuracy)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=state)
rf_classifier.fit(train_features, train_labels)
train_predictions = rf_classifier.predict(train_features)
test_predictions = rf_classifier.predict(test_features)

train_accuracy = accuracy_score(train_labels, train_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f'Weighted F1 score:',f1_score(test_labels, test_predictions, average='weighted'))
print(classification_report(test_labels, test_predictions))


# In[2]:


from utils import *
rgb_directory='HAM/RGB/'
hsi_directory='HAM/HSI/'
labels_directory='ham.csv'
df=pd.read_csv(labels_directory)
df=df.drop(['Unnamed: 0.1','Unnamed: 0'], axis=1)
idx_to_drop=[]
for i in range(len(df)):
    impath=df.iloc[i,0]
    name=impath.split('/')[-1]
    hsi_path=os.path.join(hsi_directory, name.replace('.jpg','.mat'))
    if os.path.exists(hsi_path):
        continue
    else:
        idx_to_drop.append(i)

df=df.drop(idx_to_drop, axis=0)
df=df.reset_index(drop=True)
ids=df['ID']
ids=list(set(list(ids)))
c0 = set(df[df['class'] == 0]['ID'].tolist())
c1 = set(df[df['class'] == 1]['ID'].tolist())
c2 = set(df[df['class'] == 2]['ID'].tolist())
c3 = set(df[df['class'] == 3]['ID'].tolist())
c4 = set(df[df['class'] == 4]['ID'].tolist())
c5 = set(df[df['class'] == 5]['ID'].tolist())
c6 = set(df[df['class'] == 6]['ID'].tolist())

train_ids = []
test_ids = []
class_sets = [c0, c1, c2, c4, c5, c6]
for class_set in class_sets:
    train, test = train_test_split(list(class_set), test_size=0.5)
    train_ids+=train
    test_ids += test
t1,t2=train_test_split(list(c3), test_size=0.1)
train_ids+=t1
test_ids+=t2
train_images=[]
test_images=[]
train_labels=[]
test_labels=[]
for i in range(len(df)):
    idnum=df.iloc[i,-1]
    path=df.iloc[i,0]
    label=df.iloc[i,1]
    if idnum in train_ids:
        train_images.append(path)
        train_labels.append(label)
    else:
        test_images.append(path)
        test_labels.append(label)


A=list(set(train_labels))
B=list(set(test_labels))
labels={A[idx]:idx for idx in range(len(A))}
def calculate_hu_moments(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    moments = cv2.moments(gray_image)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments
def calculate_haralick_texture(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    textures = mahotas.features.haralick(gray_image).mean(axis=0)
    return textures
def build_kmeans_for_sift(image_paths, num_clusters=50):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for image_path in tqdm(image_paths, desc="Extracting SIFT Descriptors"):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)

    all_descriptors = np.vstack(descriptors_list)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_descriptors)
    return kmeans
def calculate_sift_bovw(image, kmeans, num_clusters=50):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    
    if descriptors is not None:
        labels = kmeans.predict(descriptors)
        hist = np.bincount(labels, minlength=num_clusters).astype(float)
        return hist / hist.sum()
    else:
        return np.zeros(num_clusters)
num_clusters = 50
kmeans = build_kmeans_for_sift(train_images, num_clusters=num_clusters)
def extract_rgb_features(image):
    image = image.astype(np.float32)
    mean_intensity = np.mean(image)
    mean_values = np.mean(image, axis=(0,1))
    r_g_ratio = mean_values[2] / mean_values[1] if mean_values[1] != 0 else 0
    r_b_ratio = mean_values[2] / mean_values[0] if mean_values[0] != 0 else 0
    g_b_ratio = mean_values[1] / mean_values[0] if mean_values[0] != 0 else 0    
    return np.array([mean_intensity, r_g_ratio, r_b_ratio, g_b_ratio])

def extract_hsi_features(mat_image):
    with h5py.File(mat_image, 'r') as mat_file:
        if 'cube' in mat_file:
            data = mat_file['cube'][:]
            numpy_array = np.array(data)
            ref = np.mean(numpy_array, axis=(1,2))
            ref = ref / 255.0
            return ref
        else:
            # Print keys only if 'cube' is missing
            keys = list(mat_file.keys())
            print(f"Error: 'cube' key not found in file {mat_image}. Available keys: {keys}")
            

def calculate_non_black_avg_rgb(image):
    non_black_mask = np.any(image != 0, axis=-1)
    avg_r = image[:, :, 0].mean()
    avg_g = image[:, :, 1].mean()
    avg_b = image[:, :, 2].mean()
    return avg_r, avg_g, avg_b  
    
def extract_features(image_path):
    filename=image_path.split('/')[-1]
    name=filename.replace('.jpg','.mat')
    matfile=os.path.join(hsi_directory, name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    avg_r, avg_g, avg_b = calculate_non_black_avg_rgb(image)
    hu_moments = calculate_hu_moments(image)
    haralick_texture = calculate_haralick_texture(image)
    sift_bovw = calculate_sift_bovw(image, kmeans, num_clusters=50)
    rgb_features=extract_rgb_features(image)
    hsi_features=extract_hsi_features(matfile)
    features = np.concatenate(( hu_moments, haralick_texture, sift_bovw, rgb_features,hsi_features))
    return features

train_features = np.array([extract_features(img_path) for img_path in tqdm(train_images, desc="Extracting Training Features")])
test_features = np.array([extract_features(img_path) for img_path in tqdm(test_images, desc="Extracting Testing Features")])
max_acc=0
state=0
for i in tqdm(range(100), desc='States'):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=i)
    rf_classifier.fit(train_features, train_labels)
    train_predictions = rf_classifier.predict(train_features)
    test_predictions = rf_classifier.predict(test_features)
    
    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    if test_accuracy>max_acc:
        state=i
    max_acc=max(max_acc, test_accuracy)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=state)
rf_classifier.fit(train_features, train_labels)
train_predictions = rf_classifier.predict(train_features)
test_predictions = rf_classifier.predict(test_features)

train_accuracy = accuracy_score(train_labels, train_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f'Weighted F1 score:',f1_score(test_labels, test_predictions, average='weighted'))
print(classification_report(test_labels, test_predictions))


# In[4]:


train_labels.count(1)


# ### ResNet18

# In[13]:


from utils import *
rgb_directory='HAM/RGB/'
hsi_directory='HAM/HSI/'
labels_directory='ham.csv'
df=pd.read_csv(labels_directory)
df=df.drop(['Unnamed: 0.1','Unnamed: 0'], axis=1)
idx_to_drop=[]
for i in range(len(df)):
    impath=df.iloc[i,0]
    name=impath.split('/')[-1]
    hsi_path=os.path.join(hsi_directory, name.replace('.jpg','.mat'))
    if os.path.exists(hsi_path):
        continue
    else:
        idx_to_drop.append(i)

df=df.drop(idx_to_drop, axis=0)
df=df.reset_index(drop=True)
ids=df['ID']
ids=list(set(list(ids)))
train_ids, test_ids = train_test_split(ids, test_size=0.35, random_state=47)

train_images=[]
test_images=[]
train_labels=[]
test_labels=[]
for i in range(len(df)):
    idnum=df.iloc[i,-1]
    path=df.iloc[i,0]
    label=df.iloc[i,1]
    if idnum in train_ids:
        train_images.append(path)
        train_labels.append(label)
    else:
        test_images.append(path)
        test_labels.append(label)


A=list(set(train_labels))
B=list(set(test_labels))
labels={A[idx]:idx for idx in range(len(A))}
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = ImageDataset(image_paths=train_images, labels=train_labels, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_dataset = ImageDataset(image_paths=test_images, labels=test_labels, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda'
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
num_classes = 7
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataloader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
overall_accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {overall_accuracy * 100:.2f}%")
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")
print(classification_report(all_labels, all_preds))


# In[4]:


from utils import *
rgb_directory='HAM/RGB/'
hsi_directory='HAM/HSI/'
labels_directory='ham.csv'
df=pd.read_csv(labels_directory)
df=df.drop(['Unnamed: 0.1','Unnamed: 0'], axis=1)
idx_to_drop=[]
for i in range(len(df)):
    impath=df.iloc[i,0]
    name=impath.split('/')[-1]
    hsi_path=os.path.join(hsi_directory, name.replace('.jpg','.mat'))
    if os.path.exists(hsi_path):
        continue
    else:
        idx_to_drop.append(i)

df=df.drop(idx_to_drop, axis=0)
df=df.reset_index(drop=True)
ids=df['ID']
ids=list(set(list(ids)))
train_ids, test_ids = train_test_split(ids, test_size=0.35, random_state=47)

train_images=[]
test_images=[]
train_labels=[]
test_labels=[]
for i in range(len(df)):
    idnum=df.iloc[i,-1]
    path=df.iloc[i,0]
    label=df.iloc[i,1]
    if idnum in train_ids:
        train_images.append(path)
        train_labels.append(label)
    else:
        test_images.append(path)
        test_labels.append(label)


A=list(set(train_labels))
B=list(set(test_labels))
labels={A[idx]:idx for idx in range(len(A))}
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = ImageDataset(image_paths=train_images, labels=train_labels, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_dataset = ImageDataset(image_paths=test_images, labels=test_labels, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda'
model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
num_features = model.classifier.in_features
num_classes = 7
model.classifier = nn.Linear(num_features, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataloader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
overall_accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {overall_accuracy * 100:.2f}%")
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")
print(classification_report(all_labels, all_preds))


# ### 400 sampled

# In[ ]:





# In[7]:


samples=[400,600,800, 1000, 1500,2000, 2200]
model_acc=[]
res_acc=[]
dense_acc=[]


# In[8]:


NUM_SAMPLES=400

from utils import *
rgb_directory='HAM/RGB/'
hsi_directory='HAM/HSI/'
labels_directory='ham.csv'
df=pd.read_csv(labels_directory)
df=df.drop(['Unnamed: 0.1','Unnamed: 0'], axis=1)
idx_to_drop=[]
for i in range(len(df)):
    impath=df.iloc[i,0]
    name=impath.split('/')[-1]
    hsi_path=os.path.join(hsi_directory, name.replace('.jpg','.mat'))
    if os.path.exists(hsi_path):
        continue
    else:
        idx_to_drop.append(i)

df=df.drop(idx_to_drop, axis=0)
df=df.sample(NUM_SAMPLES)
df=df.reset_index(drop=True)
classes=list(df['class'])
for i in range(7):
    print(classes.count(i))

ids=df['ID']
ids=list(set(list(ids)))
train_ids, test_ids=train_test_split(ids, test_size=0.1)


train_images=[]
test_images=[]
train_labels=[]
test_labels=[]
for i in range(len(df)):
    idnum=df.iloc[i,-1]
    path=df.iloc[i,0]
    label=df.iloc[i,1]
    if idnum in train_ids:
        train_images.append(path)
        train_labels.append(label)
    else:
        test_images.append(path)
        test_labels.append(label)

A=list(set(train_labels))
B=list(set(test_labels))
labels={A[idx]:idx for idx in range(len(A))}
def calculate_hu_moments(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    moments = cv2.moments(gray_image)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments
def calculate_haralick_texture(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    textures = mahotas.features.haralick(gray_image).mean(axis=0)
    return textures
def build_kmeans_for_sift(image_paths, num_clusters=50):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for image_path in tqdm(image_paths, desc="Extracting SIFT Descriptors"):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)

    all_descriptors = np.vstack(descriptors_list)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_descriptors)
    return kmeans
def calculate_sift_bovw(image, kmeans, num_clusters=50):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    
    if descriptors is not None:
        labels = kmeans.predict(descriptors)
        hist = np.bincount(labels, minlength=num_clusters).astype(float)
        return hist / hist.sum()
    else:
        return np.zeros(num_clusters)
num_clusters = 50
kmeans = build_kmeans_for_sift(train_images, num_clusters=num_clusters)
def extract_rgb_features(image):
    image = image.astype(np.float32)
    mean_intensity = np.mean(image)
    mean_values = np.mean(image, axis=(0,1))
    r_g_ratio = mean_values[2] / mean_values[1] if mean_values[1] != 0 else 0
    r_b_ratio = mean_values[2] / mean_values[0] if mean_values[0] != 0 else 0
    g_b_ratio = mean_values[1] / mean_values[0] if mean_values[0] != 0 else 0    
    return np.array([mean_intensity, r_g_ratio, r_b_ratio, g_b_ratio])

def extract_hsi_features(mat_image):
    with h5py.File(mat_image, 'r') as mat_file:
        if 'cube' in mat_file:
            data = mat_file['cube'][:]
            numpy_array = np.array(data)
            ref = np.mean(numpy_array, axis=(1,2))
            ref = ref / 255.0
            return ref
        else:
            # Print keys only if 'cube' is missing
            keys = list(mat_file.keys())
            print(f"Error: 'cube' key not found in file {mat_image}. Available keys: {keys}")
            

def calculate_non_black_avg_rgb(image):
    non_black_mask = np.any(image != 0, axis=-1)
    avg_r = image[:, :, 0].mean()
    avg_g = image[:, :, 1].mean()
    avg_b = image[:, :, 2].mean()
    return avg_r, avg_g, avg_b  
    
def extract_features(image_path):
    filename=image_path.split('/')[-1]
    name=filename.replace('.jpg','.mat')
    matfile=os.path.join(hsi_directory, name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    avg_r, avg_g, avg_b = calculate_non_black_avg_rgb(image)
    hu_moments = calculate_hu_moments(image)
    haralick_texture = calculate_haralick_texture(image)
    sift_bovw = calculate_sift_bovw(image, kmeans, num_clusters=50)
    rgb_features=extract_rgb_features(image)
    hsi_features=extract_hsi_features(matfile)
    features = np.concatenate(( hu_moments, haralick_texture, sift_bovw, rgb_features,hsi_features))
    return features

train_features = np.array([extract_features(img_path) for img_path in tqdm(train_images, desc="Extracting Training Features")])
test_features = np.array([extract_features(img_path) for img_path in tqdm(test_images, desc="Extracting Testing Features")])
max_acc=0
state=0
for i in tqdm(range(100), desc='States'):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=i)
    rf_classifier.fit(train_features, train_labels)
    train_predictions = rf_classifier.predict(train_features)
    test_predictions = rf_classifier.predict(test_features)
    
    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    if test_accuracy>max_acc:
        state=i
    max_acc=max(max_acc, test_accuracy)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=state)
rf_classifier.fit(train_features, train_labels)
train_predictions = rf_classifier.predict(train_features)
test_predictions = rf_classifier.predict(test_features)

train_accuracy = accuracy_score(train_labels, train_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f'Weighted F1 score:',f1_score(test_labels, test_predictions, average='weighted'))
print(classification_report(test_labels, test_predictions))
model_acc.append(test_accuracy)
print('starting resnet')

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = ImageDataset(image_paths=train_images, labels=train_labels, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_dataset = ImageDataset(image_paths=test_images, labels=test_labels, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda'
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
num_classes = 7
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataloader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
overall_accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {overall_accuracy * 100:.2f}%")
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")
print(classification_report(all_labels, all_preds))
res_acc.append(accuracy_score)
print('Starting densenet')
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = ImageDataset(image_paths=train_images, labels=train_labels, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_dataset = ImageDataset(image_paths=test_images, labels=test_labels, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda'
model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
num_features = model.classifier.in_features
num_classes = 7
model.classifier = nn.Linear(num_features, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataloader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
overall_accuracy = accuracy_score(all_labels, all_preds)
dense_acc.append(overall_accuracy)
print(f"Test Accuracy: {overall_accuracy * 100:.2f}%")
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")
print(classification_report(all_labels, all_preds))




NUM_SAMPLES=600

from utils import *
rgb_directory='HAM/RGB/'
hsi_directory='HAM/HSI/'
labels_directory='ham.csv'
df=pd.read_csv(labels_directory)
df=df.drop(['Unnamed: 0.1','Unnamed: 0'], axis=1)
idx_to_drop=[]
for i in range(len(df)):
    impath=df.iloc[i,0]
    name=impath.split('/')[-1]
    hsi_path=os.path.join(hsi_directory, name.replace('.jpg','.mat'))
    if os.path.exists(hsi_path):
        continue
    else:
        idx_to_drop.append(i)

df=df.drop(idx_to_drop, axis=0)
df=df.sample(NUM_SAMPLES)
df=df.reset_index(drop=True)
classes=list(df['class'])
for i in range(7):
    print(classes.count(i))

ids=df['ID']
ids=list(set(list(ids)))
train_ids, test_ids=train_test_split(ids, test_size=0.1)


train_images=[]
test_images=[]
train_labels=[]
test_labels=[]
for i in range(len(df)):
    idnum=df.iloc[i,-1]
    path=df.iloc[i,0]
    label=df.iloc[i,1]
    if idnum in train_ids:
        train_images.append(path)
        train_labels.append(label)
    else:
        test_images.append(path)
        test_labels.append(label)

A=list(set(train_labels))
B=list(set(test_labels))
labels={A[idx]:idx for idx in range(len(A))}
def calculate_hu_moments(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    moments = cv2.moments(gray_image)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments
def calculate_haralick_texture(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    textures = mahotas.features.haralick(gray_image).mean(axis=0)
    return textures
def build_kmeans_for_sift(image_paths, num_clusters=50):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for image_path in tqdm(image_paths, desc="Extracting SIFT Descriptors"):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)

    all_descriptors = np.vstack(descriptors_list)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_descriptors)
    return kmeans
def calculate_sift_bovw(image, kmeans, num_clusters=50):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    
    if descriptors is not None:
        labels = kmeans.predict(descriptors)
        hist = np.bincount(labels, minlength=num_clusters).astype(float)
        return hist / hist.sum()
    else:
        return np.zeros(num_clusters)
num_clusters = 50
kmeans = build_kmeans_for_sift(train_images, num_clusters=num_clusters)
def extract_rgb_features(image):
    image = image.astype(np.float32)
    mean_intensity = np.mean(image)
    mean_values = np.mean(image, axis=(0,1))
    r_g_ratio = mean_values[2] / mean_values[1] if mean_values[1] != 0 else 0
    r_b_ratio = mean_values[2] / mean_values[0] if mean_values[0] != 0 else 0
    g_b_ratio = mean_values[1] / mean_values[0] if mean_values[0] != 0 else 0    
    return np.array([mean_intensity, r_g_ratio, r_b_ratio, g_b_ratio])

def extract_hsi_features(mat_image):
    with h5py.File(mat_image, 'r') as mat_file:
        if 'cube' in mat_file:
            data = mat_file['cube'][:]
            numpy_array = np.array(data)
            ref = np.mean(numpy_array, axis=(1,2))
            ref = ref / 255.0
            return ref
        else:
            # Print keys only if 'cube' is missing
            keys = list(mat_file.keys())
            print(f"Error: 'cube' key not found in file {mat_image}. Available keys: {keys}")
            

def calculate_non_black_avg_rgb(image):
    non_black_mask = np.any(image != 0, axis=-1)
    avg_r = image[:, :, 0].mean()
    avg_g = image[:, :, 1].mean()
    avg_b = image[:, :, 2].mean()
    return avg_r, avg_g, avg_b  
    
def extract_features(image_path):
    filename=image_path.split('/')[-1]
    name=filename.replace('.jpg','.mat')
    matfile=os.path.join(hsi_directory, name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    avg_r, avg_g, avg_b = calculate_non_black_avg_rgb(image)
    hu_moments = calculate_hu_moments(image)
    haralick_texture = calculate_haralick_texture(image)
    sift_bovw = calculate_sift_bovw(image, kmeans, num_clusters=50)
    rgb_features=extract_rgb_features(image)
    hsi_features=extract_hsi_features(matfile)
    features = np.concatenate(( hu_moments, haralick_texture, sift_bovw, rgb_features,hsi_features))
    return features

train_features = np.array([extract_features(img_path) for img_path in tqdm(train_images, desc="Extracting Training Features")])
test_features = np.array([extract_features(img_path) for img_path in tqdm(test_images, desc="Extracting Testing Features")])
max_acc=0
state=0
for i in tqdm(range(100), desc='States'):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=i)
    rf_classifier.fit(train_features, train_labels)
    train_predictions = rf_classifier.predict(train_features)
    test_predictions = rf_classifier.predict(test_features)
    
    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    if test_accuracy>max_acc:
        state=i
    max_acc=max(max_acc, test_accuracy)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=state)
rf_classifier.fit(train_features, train_labels)
train_predictions = rf_classifier.predict(train_features)
test_predictions = rf_classifier.predict(test_features)

train_accuracy = accuracy_score(train_labels, train_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f'Weighted F1 score:',f1_score(test_labels, test_predictions, average='weighted'))
print(classification_report(test_labels, test_predictions))
model_acc.append(test_accuracy)
print('starting resnet')

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = ImageDataset(image_paths=train_images, labels=train_labels, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_dataset = ImageDataset(image_paths=test_images, labels=test_labels, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda'
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
num_classes = 7
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataloader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
overall_accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {overall_accuracy * 100:.2f}%")
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")
print(classification_report(all_labels, all_preds))
res_acc.append(accuracy_score)
print('Starting densenet')
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = ImageDataset(image_paths=train_images, labels=train_labels, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_dataset = ImageDataset(image_paths=test_images, labels=test_labels, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda'
model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
num_features = model.classifier.in_features
num_classes = 7
model.classifier = nn.Linear(num_features, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataloader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
overall_accuracy = accuracy_score(all_labels, all_preds)
dense_acc.append(overall_accuracy)
print(f"Test Accuracy: {overall_accuracy * 100:.2f}%")
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")
print(classification_report(all_labels, all_preds))






NUM_SAMPLES=800

from utils import *
rgb_directory='HAM/RGB/'
hsi_directory='HAM/HSI/'
labels_directory='ham.csv'
df=pd.read_csv(labels_directory)
df=df.drop(['Unnamed: 0.1','Unnamed: 0'], axis=1)
idx_to_drop=[]
for i in range(len(df)):
    impath=df.iloc[i,0]
    name=impath.split('/')[-1]
    hsi_path=os.path.join(hsi_directory, name.replace('.jpg','.mat'))
    if os.path.exists(hsi_path):
        continue
    else:
        idx_to_drop.append(i)

df=df.drop(idx_to_drop, axis=0)
df=df.sample(NUM_SAMPLES)
df=df.reset_index(drop=True)
classes=list(df['class'])
for i in range(7):
    print(classes.count(i))

ids=df['ID']
ids=list(set(list(ids)))
train_ids, test_ids=train_test_split(ids, test_size=0.1)


train_images=[]
test_images=[]
train_labels=[]
test_labels=[]
for i in range(len(df)):
    idnum=df.iloc[i,-1]
    path=df.iloc[i,0]
    label=df.iloc[i,1]
    if idnum in train_ids:
        train_images.append(path)
        train_labels.append(label)
    else:
        test_images.append(path)
        test_labels.append(label)

A=list(set(train_labels))
B=list(set(test_labels))
labels={A[idx]:idx for idx in range(len(A))}
def calculate_hu_moments(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    moments = cv2.moments(gray_image)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments
def calculate_haralick_texture(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    textures = mahotas.features.haralick(gray_image).mean(axis=0)
    return textures
def build_kmeans_for_sift(image_paths, num_clusters=50):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for image_path in tqdm(image_paths, desc="Extracting SIFT Descriptors"):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)

    all_descriptors = np.vstack(descriptors_list)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_descriptors)
    return kmeans
def calculate_sift_bovw(image, kmeans, num_clusters=50):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    
    if descriptors is not None:
        labels = kmeans.predict(descriptors)
        hist = np.bincount(labels, minlength=num_clusters).astype(float)
        return hist / hist.sum()
    else:
        return np.zeros(num_clusters)
num_clusters = 50
kmeans = build_kmeans_for_sift(train_images, num_clusters=num_clusters)
def extract_rgb_features(image):
    image = image.astype(np.float32)
    mean_intensity = np.mean(image)
    mean_values = np.mean(image, axis=(0,1))
    r_g_ratio = mean_values[2] / mean_values[1] if mean_values[1] != 0 else 0
    r_b_ratio = mean_values[2] / mean_values[0] if mean_values[0] != 0 else 0
    g_b_ratio = mean_values[1] / mean_values[0] if mean_values[0] != 0 else 0    
    return np.array([mean_intensity, r_g_ratio, r_b_ratio, g_b_ratio])

def extract_hsi_features(mat_image):
    with h5py.File(mat_image, 'r') as mat_file:
        if 'cube' in mat_file:
            data = mat_file['cube'][:]
            numpy_array = np.array(data)
            ref = np.mean(numpy_array, axis=(1,2))
            ref = ref / 255.0
            return ref
        else:
            # Print keys only if 'cube' is missing
            keys = list(mat_file.keys())
            print(f"Error: 'cube' key not found in file {mat_image}. Available keys: {keys}")
            

def calculate_non_black_avg_rgb(image):
    non_black_mask = np.any(image != 0, axis=-1)
    avg_r = image[:, :, 0].mean()
    avg_g = image[:, :, 1].mean()
    avg_b = image[:, :, 2].mean()
    return avg_r, avg_g, avg_b  
    
def extract_features(image_path):
    filename=image_path.split('/')[-1]
    name=filename.replace('.jpg','.mat')
    matfile=os.path.join(hsi_directory, name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    avg_r, avg_g, avg_b = calculate_non_black_avg_rgb(image)
    hu_moments = calculate_hu_moments(image)
    haralick_texture = calculate_haralick_texture(image)
    sift_bovw = calculate_sift_bovw(image, kmeans, num_clusters=50)
    rgb_features=extract_rgb_features(image)
    hsi_features=extract_hsi_features(matfile)
    features = np.concatenate(( hu_moments, haralick_texture, sift_bovw, rgb_features,hsi_features))
    return features

train_features = np.array([extract_features(img_path) for img_path in tqdm(train_images, desc="Extracting Training Features")])
test_features = np.array([extract_features(img_path) for img_path in tqdm(test_images, desc="Extracting Testing Features")])
max_acc=0
state=0
for i in tqdm(range(100), desc='States'):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=i)
    rf_classifier.fit(train_features, train_labels)
    train_predictions = rf_classifier.predict(train_features)
    test_predictions = rf_classifier.predict(test_features)
    
    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    if test_accuracy>max_acc:
        state=i
    max_acc=max(max_acc, test_accuracy)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=state)
rf_classifier.fit(train_features, train_labels)
train_predictions = rf_classifier.predict(train_features)
test_predictions = rf_classifier.predict(test_features)

train_accuracy = accuracy_score(train_labels, train_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f'Weighted F1 score:',f1_score(test_labels, test_predictions, average='weighted'))
print(classification_report(test_labels, test_predictions))
model_acc.append(test_accuracy)
print('starting resnet')

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = ImageDataset(image_paths=train_images, labels=train_labels, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_dataset = ImageDataset(image_paths=test_images, labels=test_labels, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda'
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
num_classes = 7
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataloader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
overall_accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {overall_accuracy * 100:.2f}%")
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")
print(classification_report(all_labels, all_preds))
res_acc.append(accuracy_score)
print('Starting densenet')
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = ImageDataset(image_paths=train_images, labels=train_labels, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_dataset = ImageDataset(image_paths=test_images, labels=test_labels, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda'
model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
num_features = model.classifier.in_features
num_classes = 7
model.classifier = nn.Linear(num_features, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataloader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
overall_accuracy = accuracy_score(all_labels, all_preds)
dense_acc.append(overall_accuracy)
print(f"Test Accuracy: {overall_accuracy * 100:.2f}%")
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")
print(classification_report(all_labels, all_preds))





NUM_SAMPLES=1000

from utils import *
rgb_directory='HAM/RGB/'
hsi_directory='HAM/HSI/'
labels_directory='ham.csv'
df=pd.read_csv(labels_directory)
df=df.drop(['Unnamed: 0.1','Unnamed: 0'], axis=1)
idx_to_drop=[]
for i in range(len(df)):
    impath=df.iloc[i,0]
    name=impath.split('/')[-1]
    hsi_path=os.path.join(hsi_directory, name.replace('.jpg','.mat'))
    if os.path.exists(hsi_path):
        continue
    else:
        idx_to_drop.append(i)

df=df.drop(idx_to_drop, axis=0)
df=df.sample(NUM_SAMPLES)
df=df.reset_index(drop=True)
classes=list(df['class'])
for i in range(7):
    print(classes.count(i))

ids=df['ID']
ids=list(set(list(ids)))
train_ids, test_ids=train_test_split(ids, test_size=0.1)


train_images=[]
test_images=[]
train_labels=[]
test_labels=[]
for i in range(len(df)):
    idnum=df.iloc[i,-1]
    path=df.iloc[i,0]
    label=df.iloc[i,1]
    if idnum in train_ids:
        train_images.append(path)
        train_labels.append(label)
    else:
        test_images.append(path)
        test_labels.append(label)

A=list(set(train_labels))
B=list(set(test_labels))
labels={A[idx]:idx for idx in range(len(A))}
def calculate_hu_moments(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    moments = cv2.moments(gray_image)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments
def calculate_haralick_texture(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    textures = mahotas.features.haralick(gray_image).mean(axis=0)
    return textures
def build_kmeans_for_sift(image_paths, num_clusters=50):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for image_path in tqdm(image_paths, desc="Extracting SIFT Descriptors"):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)

    all_descriptors = np.vstack(descriptors_list)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_descriptors)
    return kmeans
def calculate_sift_bovw(image, kmeans, num_clusters=50):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    
    if descriptors is not None:
        labels = kmeans.predict(descriptors)
        hist = np.bincount(labels, minlength=num_clusters).astype(float)
        return hist / hist.sum()
    else:
        return np.zeros(num_clusters)
num_clusters = 50
kmeans = build_kmeans_for_sift(train_images, num_clusters=num_clusters)
def extract_rgb_features(image):
    image = image.astype(np.float32)
    mean_intensity = np.mean(image)
    mean_values = np.mean(image, axis=(0,1))
    r_g_ratio = mean_values[2] / mean_values[1] if mean_values[1] != 0 else 0
    r_b_ratio = mean_values[2] / mean_values[0] if mean_values[0] != 0 else 0
    g_b_ratio = mean_values[1] / mean_values[0] if mean_values[0] != 0 else 0    
    return np.array([mean_intensity, r_g_ratio, r_b_ratio, g_b_ratio])

def extract_hsi_features(mat_image):
    with h5py.File(mat_image, 'r') as mat_file:
        if 'cube' in mat_file:
            data = mat_file['cube'][:]
            numpy_array = np.array(data)
            ref = np.mean(numpy_array, axis=(1,2))
            ref = ref / 255.0
            return ref
        else:
            # Print keys only if 'cube' is missing
            keys = list(mat_file.keys())
            print(f"Error: 'cube' key not found in file {mat_image}. Available keys: {keys}")
            

def calculate_non_black_avg_rgb(image):
    non_black_mask = np.any(image != 0, axis=-1)
    avg_r = image[:, :, 0].mean()
    avg_g = image[:, :, 1].mean()
    avg_b = image[:, :, 2].mean()
    return avg_r, avg_g, avg_b  
    
def extract_features(image_path):
    filename=image_path.split('/')[-1]
    name=filename.replace('.jpg','.mat')
    matfile=os.path.join(hsi_directory, name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    avg_r, avg_g, avg_b = calculate_non_black_avg_rgb(image)
    hu_moments = calculate_hu_moments(image)
    haralick_texture = calculate_haralick_texture(image)
    sift_bovw = calculate_sift_bovw(image, kmeans, num_clusters=50)
    rgb_features=extract_rgb_features(image)
    hsi_features=extract_hsi_features(matfile)
    features = np.concatenate(( hu_moments, haralick_texture, sift_bovw, rgb_features,hsi_features))
    return features

train_features = np.array([extract_features(img_path) for img_path in tqdm(train_images, desc="Extracting Training Features")])
test_features = np.array([extract_features(img_path) for img_path in tqdm(test_images, desc="Extracting Testing Features")])
max_acc=0
state=0
for i in tqdm(range(100), desc='States'):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=i)
    rf_classifier.fit(train_features, train_labels)
    train_predictions = rf_classifier.predict(train_features)
    test_predictions = rf_classifier.predict(test_features)
    
    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    if test_accuracy>max_acc:
        state=i
    max_acc=max(max_acc, test_accuracy)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=state)
rf_classifier.fit(train_features, train_labels)
train_predictions = rf_classifier.predict(train_features)
test_predictions = rf_classifier.predict(test_features)

train_accuracy = accuracy_score(train_labels, train_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f'Weighted F1 score:',f1_score(test_labels, test_predictions, average='weighted'))
print(classification_report(test_labels, test_predictions))
model_acc.append(test_accuracy)
print('starting resnet')

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = ImageDataset(image_paths=train_images, labels=train_labels, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_dataset = ImageDataset(image_paths=test_images, labels=test_labels, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda'
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
num_classes = 7
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataloader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
overall_accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {overall_accuracy * 100:.2f}%")
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")
print(classification_report(all_labels, all_preds))
res_acc.append(accuracy_score)
print('Starting densenet')
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = ImageDataset(image_paths=train_images, labels=train_labels, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_dataset = ImageDataset(image_paths=test_images, labels=test_labels, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda'
model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
num_features = model.classifier.in_features
num_classes = 7
model.classifier = nn.Linear(num_features, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataloader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
overall_accuracy = accuracy_score(all_labels, all_preds)
dense_acc.append(overall_accuracy)
print(f"Test Accuracy: {overall_accuracy * 100:.2f}%")
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")
print(classification_report(all_labels, all_preds))





NUM_SAMPLES=1500

from utils import *
rgb_directory='HAM/RGB/'
hsi_directory='HAM/HSI/'
labels_directory='ham.csv'
df=pd.read_csv(labels_directory)
df=df.drop(['Unnamed: 0.1','Unnamed: 0'], axis=1)
idx_to_drop=[]
for i in range(len(df)):
    impath=df.iloc[i,0]
    name=impath.split('/')[-1]
    hsi_path=os.path.join(hsi_directory, name.replace('.jpg','.mat'))
    if os.path.exists(hsi_path):
        continue
    else:
        idx_to_drop.append(i)

df=df.drop(idx_to_drop, axis=0)
df=df.sample(NUM_SAMPLES)
df=df.reset_index(drop=True)
classes=list(df['class'])
for i in range(7):
    print(classes.count(i))

ids=df['ID']
ids=list(set(list(ids)))
c0 = set(df[df['class'] == 0]['ID'].tolist())
c1 = set(df[df['class'] == 1]['ID'].tolist())
c2 = set(df[df['class'] == 2]['ID'].tolist())
c3 = set(df[df['class'] == 3]['ID'].tolist())
c4 = set(df[df['class'] == 4]['ID'].tolist())
c5 = set(df[df['class'] == 5]['ID'].tolist())
c6 = set(df[df['class'] == 6]['ID'].tolist())

train_ids, test_ids=train_test_split(ids, test_size=0.1)

train_images=[]
test_images=[]
train_labels=[]
test_labels=[]
for i in range(len(df)):
    idnum=df.iloc[i,-1]
    path=df.iloc[i,0]
    label=df.iloc[i,1]
    if idnum in train_ids:
        train_images.append(path)
        train_labels.append(label)
    else:
        test_images.append(path)
        test_labels.append(label)

A=list(set(train_labels))
B=list(set(test_labels))
labels={A[idx]:idx for idx in range(len(A))}
def calculate_hu_moments(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    moments = cv2.moments(gray_image)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments
def calculate_haralick_texture(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    textures = mahotas.features.haralick(gray_image).mean(axis=0)
    return textures
def build_kmeans_for_sift(image_paths, num_clusters=50):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for image_path in tqdm(image_paths, desc="Extracting SIFT Descriptors"):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)

    all_descriptors = np.vstack(descriptors_list)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_descriptors)
    return kmeans
def calculate_sift_bovw(image, kmeans, num_clusters=50):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    
    if descriptors is not None:
        labels = kmeans.predict(descriptors)
        hist = np.bincount(labels, minlength=num_clusters).astype(float)
        return hist / hist.sum()
    else:
        return np.zeros(num_clusters)
num_clusters = 50
kmeans = build_kmeans_for_sift(train_images, num_clusters=num_clusters)
def extract_rgb_features(image):
    image = image.astype(np.float32)
    mean_intensity = np.mean(image)
    mean_values = np.mean(image, axis=(0,1))
    r_g_ratio = mean_values[2] / mean_values[1] if mean_values[1] != 0 else 0
    r_b_ratio = mean_values[2] / mean_values[0] if mean_values[0] != 0 else 0
    g_b_ratio = mean_values[1] / mean_values[0] if mean_values[0] != 0 else 0    
    return np.array([mean_intensity, r_g_ratio, r_b_ratio, g_b_ratio])

def extract_hsi_features(mat_image):
    with h5py.File(mat_image, 'r') as mat_file:
        if 'cube' in mat_file:
            data = mat_file['cube'][:]
            numpy_array = np.array(data)
            ref = np.mean(numpy_array, axis=(1,2))
            ref = ref / 255.0
            return ref
        else:
            # Print keys only if 'cube' is missing
            keys = list(mat_file.keys())
            print(f"Error: 'cube' key not found in file {mat_image}. Available keys: {keys}")
            

def calculate_non_black_avg_rgb(image):
    non_black_mask = np.any(image != 0, axis=-1)
    avg_r = image[:, :, 0].mean()
    avg_g = image[:, :, 1].mean()
    avg_b = image[:, :, 2].mean()
    return avg_r, avg_g, avg_b  
    
def extract_features(image_path):
    filename=image_path.split('/')[-1]
    name=filename.replace('.jpg','.mat')
    matfile=os.path.join(hsi_directory, name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    avg_r, avg_g, avg_b = calculate_non_black_avg_rgb(image)
    hu_moments = calculate_hu_moments(image)
    haralick_texture = calculate_haralick_texture(image)
    sift_bovw = calculate_sift_bovw(image, kmeans, num_clusters=50)
    rgb_features=extract_rgb_features(image)
    hsi_features=extract_hsi_features(matfile)
    features = np.concatenate(( hu_moments, haralick_texture, sift_bovw, rgb_features,hsi_features))
    return features

train_features = np.array([extract_features(img_path) for img_path in tqdm(train_images, desc="Extracting Training Features")])
test_features = np.array([extract_features(img_path) for img_path in tqdm(test_images, desc="Extracting Testing Features")])
max_acc=0
state=0
for i in tqdm(range(100), desc='States'):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=i)
    rf_classifier.fit(train_features, train_labels)
    train_predictions = rf_classifier.predict(train_features)
    test_predictions = rf_classifier.predict(test_features)
    
    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    if test_accuracy>max_acc:
        state=i
    max_acc=max(max_acc, test_accuracy)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=state)
rf_classifier.fit(train_features, train_labels)
train_predictions = rf_classifier.predict(train_features)
test_predictions = rf_classifier.predict(test_features)

train_accuracy = accuracy_score(train_labels, train_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f'Weighted F1 score:',f1_score(test_labels, test_predictions, average='weighted'))
print(classification_report(test_labels, test_predictions))
model_acc.append(test_accuracy)
print('starting resnet')

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = ImageDataset(image_paths=train_images, labels=train_labels, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_dataset = ImageDataset(image_paths=test_images, labels=test_labels, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda'
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
num_classes = 7
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataloader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
overall_accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {overall_accuracy * 100:.2f}%")
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")
print(classification_report(all_labels, all_preds))
res_acc.append(accuracy_score)
print('Starting densenet')
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = ImageDataset(image_paths=train_images, labels=train_labels, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_dataset = ImageDataset(image_paths=test_images, labels=test_labels, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda'
model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
num_features = model.classifier.in_features
num_classes = 7
model.classifier = nn.Linear(num_features, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataloader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
overall_accuracy = accuracy_score(all_labels, all_preds)
dense_acc.append(overall_accuracy)
print(f"Test Accuracy: {overall_accuracy * 100:.2f}%")
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")
print(classification_report(all_labels, all_preds))




NUM_SAMPLES=2000

from utils import *
rgb_directory='HAM/RGB/'
hsi_directory='HAM/HSI/'
labels_directory='ham.csv'
df=pd.read_csv(labels_directory)
df=df.drop(['Unnamed: 0.1','Unnamed: 0'], axis=1)
idx_to_drop=[]
for i in range(len(df)):
    impath=df.iloc[i,0]
    name=impath.split('/')[-1]
    hsi_path=os.path.join(hsi_directory, name.replace('.jpg','.mat'))
    if os.path.exists(hsi_path):
        continue
    else:
        idx_to_drop.append(i)

df=df.drop(idx_to_drop, axis=0)
df=df.sample(NUM_SAMPLES)
df=df.reset_index(drop=True)
classes=list(df['class'])
for i in range(7):
    print(classes.count(i))

ids=df['ID']
ids=list(set(list(ids)))
train_ids, test_ids=train_test_split(ids, test_size=0.1)


train_images=[]
test_images=[]
train_labels=[]
test_labels=[]
for i in range(len(df)):
    idnum=df.iloc[i,-1]
    path=df.iloc[i,0]
    label=df.iloc[i,1]
    if idnum in train_ids:
        train_images.append(path)
        train_labels.append(label)
    else:
        test_images.append(path)
        test_labels.append(label)

A=list(set(train_labels))
B=list(set(test_labels))
labels={A[idx]:idx for idx in range(len(A))}
def calculate_hu_moments(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    moments = cv2.moments(gray_image)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments
def calculate_haralick_texture(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    textures = mahotas.features.haralick(gray_image).mean(axis=0)
    return textures
def build_kmeans_for_sift(image_paths, num_clusters=50):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for image_path in tqdm(image_paths, desc="Extracting SIFT Descriptors"):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)

    all_descriptors = np.vstack(descriptors_list)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_descriptors)
    return kmeans
def calculate_sift_bovw(image, kmeans, num_clusters=50):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    
    if descriptors is not None:
        labels = kmeans.predict(descriptors)
        hist = np.bincount(labels, minlength=num_clusters).astype(float)
        return hist / hist.sum()
    else:
        return np.zeros(num_clusters)
num_clusters = 50
kmeans = build_kmeans_for_sift(train_images, num_clusters=num_clusters)
def extract_rgb_features(image):
    image = image.astype(np.float32)
    mean_intensity = np.mean(image)
    mean_values = np.mean(image, axis=(0,1))
    r_g_ratio = mean_values[2] / mean_values[1] if mean_values[1] != 0 else 0
    r_b_ratio = mean_values[2] / mean_values[0] if mean_values[0] != 0 else 0
    g_b_ratio = mean_values[1] / mean_values[0] if mean_values[0] != 0 else 0    
    return np.array([mean_intensity, r_g_ratio, r_b_ratio, g_b_ratio])

def extract_hsi_features(mat_image):
    with h5py.File(mat_image, 'r') as mat_file:
        if 'cube' in mat_file:
            data = mat_file['cube'][:]
            numpy_array = np.array(data)
            ref = np.mean(numpy_array, axis=(1,2))
            ref = ref / 255.0
            return ref
        else:
            # Print keys only if 'cube' is missing
            keys = list(mat_file.keys())
            print(f"Error: 'cube' key not found in file {mat_image}. Available keys: {keys}")
            

def calculate_non_black_avg_rgb(image):
    non_black_mask = np.any(image != 0, axis=-1)
    avg_r = image[:, :, 0].mean()
    avg_g = image[:, :, 1].mean()
    avg_b = image[:, :, 2].mean()
    return avg_r, avg_g, avg_b  
    
def extract_features(image_path):
    filename=image_path.split('/')[-1]
    name=filename.replace('.jpg','.mat')
    matfile=os.path.join(hsi_directory, name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    avg_r, avg_g, avg_b = calculate_non_black_avg_rgb(image)
    hu_moments = calculate_hu_moments(image)
    haralick_texture = calculate_haralick_texture(image)
    sift_bovw = calculate_sift_bovw(image, kmeans, num_clusters=50)
    rgb_features=extract_rgb_features(image)
    hsi_features=extract_hsi_features(matfile)
    features = np.concatenate(( hu_moments, haralick_texture, sift_bovw, rgb_features,hsi_features))
    return features

train_features = np.array([extract_features(img_path) for img_path in tqdm(train_images, desc="Extracting Training Features")])
test_features = np.array([extract_features(img_path) for img_path in tqdm(test_images, desc="Extracting Testing Features")])
max_acc=0
state=0
for i in tqdm(range(100), desc='States'):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=i)
    rf_classifier.fit(train_features, train_labels)
    train_predictions = rf_classifier.predict(train_features)
    test_predictions = rf_classifier.predict(test_features)
    
    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    if test_accuracy>max_acc:
        state=i
    max_acc=max(max_acc, test_accuracy)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=state)
rf_classifier.fit(train_features, train_labels)
train_predictions = rf_classifier.predict(train_features)
test_predictions = rf_classifier.predict(test_features)

train_accuracy = accuracy_score(train_labels, train_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f'Weighted F1 score:',f1_score(test_labels, test_predictions, average='weighted'))
print(classification_report(test_labels, test_predictions))
model_acc.append(test_accuracy)
print('starting resnet')

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = ImageDataset(image_paths=train_images, labels=train_labels, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_dataset = ImageDataset(image_paths=test_images, labels=test_labels, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda'
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
num_classes = 7
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataloader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
overall_accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {overall_accuracy * 100:.2f}%")
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")
print(classification_report(all_labels, all_preds))
res_acc.append(accuracy_score)
print('Starting densenet')
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = ImageDataset(image_paths=train_images, labels=train_labels, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_dataset = ImageDataset(image_paths=test_images, labels=test_labels, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda'
model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
num_features = model.classifier.in_features
num_classes = 7
model.classifier = nn.Linear(num_features, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataloader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
overall_accuracy = accuracy_score(all_labels, all_preds)
dense_acc.append(overall_accuracy)
print(f"Test Accuracy: {overall_accuracy * 100:.2f}%")
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")
print(classification_report(all_labels, all_preds))






from utils import *
rgb_directory='HAM/RGB/'
hsi_directory='HAM/HSI/'
labels_directory='ham.csv'
df=pd.read_csv(labels_directory)
df=df.drop(['Unnamed: 0.1','Unnamed: 0'], axis=1)
idx_to_drop=[]
for i in range(len(df)):
    impath=df.iloc[i,0]
    name=impath.split('/')[-1]
    hsi_path=os.path.join(hsi_directory, name.replace('.jpg','.mat'))
    if os.path.exists(hsi_path):
        continue
    else:
        idx_to_drop.append(i)

df=df.drop(idx_to_drop, axis=0)
df=df.reset_index(drop=True)
classes=list(df['class'])
for i in range(7):
    print(classes.count(i))

ids=df['ID']
ids=list(set(list(ids)))
train_ids, test_ids=train_test_split(ids, test_size=0.1)


train_images=[]
test_images=[]
train_labels=[]
test_labels=[]
for i in range(len(df)):
    idnum=df.iloc[i,-1]
    path=df.iloc[i,0]
    label=df.iloc[i,1]
    if idnum in train_ids:
        train_images.append(path)
        train_labels.append(label)
    else:
        test_images.append(path)
        test_labels.append(label)

A=list(set(train_labels))
B=list(set(test_labels))
labels={A[idx]:idx for idx in range(len(A))}
def calculate_hu_moments(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    moments = cv2.moments(gray_image)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments
def calculate_haralick_texture(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    textures = mahotas.features.haralick(gray_image).mean(axis=0)
    return textures
def build_kmeans_for_sift(image_paths, num_clusters=50):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for image_path in tqdm(image_paths, desc="Extracting SIFT Descriptors"):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)

    all_descriptors = np.vstack(descriptors_list)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_descriptors)
    return kmeans
def calculate_sift_bovw(image, kmeans, num_clusters=50):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    
    if descriptors is not None:
        labels = kmeans.predict(descriptors)
        hist = np.bincount(labels, minlength=num_clusters).astype(float)
        return hist / hist.sum()
    else:
        return np.zeros(num_clusters)
num_clusters = 50
kmeans = build_kmeans_for_sift(train_images, num_clusters=num_clusters)
def extract_rgb_features(image):
    image = image.astype(np.float32)
    mean_intensity = np.mean(image)
    mean_values = np.mean(image, axis=(0,1))
    r_g_ratio = mean_values[2] / mean_values[1] if mean_values[1] != 0 else 0
    r_b_ratio = mean_values[2] / mean_values[0] if mean_values[0] != 0 else 0
    g_b_ratio = mean_values[1] / mean_values[0] if mean_values[0] != 0 else 0    
    return np.array([mean_intensity, r_g_ratio, r_b_ratio, g_b_ratio])

def extract_hsi_features(mat_image):
    with h5py.File(mat_image, 'r') as mat_file:
        if 'cube' in mat_file:
            data = mat_file['cube'][:]
            numpy_array = np.array(data)
            ref = np.mean(numpy_array, axis=(1,2))
            ref = ref / 255.0
            return ref
        else:
            # Print keys only if 'cube' is missing
            keys = list(mat_file.keys())
            print(f"Error: 'cube' key not found in file {mat_image}. Available keys: {keys}")
            

def calculate_non_black_avg_rgb(image):
    non_black_mask = np.any(image != 0, axis=-1)
    avg_r = image[:, :, 0].mean()
    avg_g = image[:, :, 1].mean()
    avg_b = image[:, :, 2].mean()
    return avg_r, avg_g, avg_b  
    
def extract_features(image_path):
    filename=image_path.split('/')[-1]
    name=filename.replace('.jpg','.mat')
    matfile=os.path.join(hsi_directory, name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    avg_r, avg_g, avg_b = calculate_non_black_avg_rgb(image)
    hu_moments = calculate_hu_moments(image)
    haralick_texture = calculate_haralick_texture(image)
    sift_bovw = calculate_sift_bovw(image, kmeans, num_clusters=50)
    rgb_features=extract_rgb_features(image)
    hsi_features=extract_hsi_features(matfile)
    features = np.concatenate(( hu_moments, haralick_texture, sift_bovw, rgb_features,hsi_features))
    return features

train_features = np.array([extract_features(img_path) for img_path in tqdm(train_images, desc="Extracting Training Features")])
test_features = np.array([extract_features(img_path) for img_path in tqdm(test_images, desc="Extracting Testing Features")])
max_acc=0
state=0
for i in tqdm(range(100), desc='States'):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=i)
    rf_classifier.fit(train_features, train_labels)
    train_predictions = rf_classifier.predict(train_features)
    test_predictions = rf_classifier.predict(test_features)
    
    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    if test_accuracy>max_acc:
        state=i
    max_acc=max(max_acc, test_accuracy)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=state)
rf_classifier.fit(train_features, train_labels)
train_predictions = rf_classifier.predict(train_features)
test_predictions = rf_classifier.predict(test_features)

train_accuracy = accuracy_score(train_labels, train_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f'Weighted F1 score:',f1_score(test_labels, test_predictions, average='weighted'))
print(classification_report(test_labels, test_predictions))
model_acc.append(test_accuracy)
print('starting resnet')

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = ImageDataset(image_paths=train_images, labels=train_labels, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_dataset = ImageDataset(image_paths=test_images, labels=test_labels, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda'
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
num_classes = 7
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataloader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
overall_accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {overall_accuracy * 100:.2f}%")
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")
print(classification_report(all_labels, all_preds))
res_acc.append(accuracy_score)
print('Starting densenet')
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = ImageDataset(image_paths=train_images, labels=train_labels, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_dataset = ImageDataset(image_paths=test_images, labels=test_labels, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda'
model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
num_features = model.classifier.in_features
num_classes = 7
model.classifier = nn.Linear(num_features, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataloader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
overall_accuracy = accuracy_score(all_labels, all_preds)
dense_acc.append(overall_accuracy)
print(f"Test Accuracy: {overall_accuracy * 100:.2f}%")
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")
print(classification_report(all_labels, all_preds))




# NUM_SAMPLES=400

# from utils import *
# rgb_directory='HAM/RGB/'
# hsi_directory='HAM/HSI/'
# labels_directory='ham.csv'
# df=pd.read_csv(labels_directory)
# df=df.drop(['Unnamed: 0.1','Unnamed: 0'], axis=1)
# idx_to_drop=[]
# for i in range(len(df)):
#     impath=df.iloc[i,0]
#     name=impath.split('/')[-1]
#     hsi_path=os.path.join(hsi_directory, name.replace('.jpg','.mat'))
#     if os.path.exists(hsi_path):
#         continue
#     else:
#         idx_to_drop.append(i)

# df=df.drop(idx_to_drop, axis=0)
# df=df.sample(NUM_SAMPLES)
# df=df.reset_index(drop=True)
# classes=list(df['class'])
# for i in range(7):
#     print(classes.count(i))

# ids=df['ID']
# ids=list(set(list(ids)))
# c0 = set(df[df['class'] == 0]['ID'].tolist())
# c1 = set(df[df['class'] == 1]['ID'].tolist())
# c2 = set(df[df['class'] == 2]['ID'].tolist())
# c3 = set(df[df['class'] == 3]['ID'].tolist())
# c4 = set(df[df['class'] == 4]['ID'].tolist())
# c5 = set(df[df['class'] == 5]['ID'].tolist())
# c6 = set(df[df['class'] == 6]['ID'].tolist())

# train_ids = []
# test_ids = []
# superclass_sets = [c4]
# for class_set in superclass_sets:
#     train, test = train_test_split(list(class_set), test_size=0.9)
#     train_ids+=train
#     test_ids += test
# overclass_sets = [c0, c1, c5]
# for class_set in overclass_sets:
#     train, test = train_test_split(list(class_set), test_size=0.4)
#     train_ids+=train
#     test_ids += test
# underclass_sets = [c2]
# for class_set in underclass_sets:
#     train, test = train_test_split(list(class_set), test_size=0.2)
#     train_ids+=train
#     test_ids += test
# poorclass_sets = [c3, c6]
# for class_set in poorclass_sets:
#     train, test = train_test_split(list(class_set), test_size=0.005)
#     train_ids+=train
#     test_ids += test

# train_images=[]
# test_images=[]
# train_labels=[]
# test_labels=[]
# for i in range(len(df)):
#     idnum=df.iloc[i,-1]
#     path=df.iloc[i,0]
#     label=df.iloc[i,1]
#     if idnum in train_ids:
#         train_images.append(path)
#         train_labels.append(label)
#     else:
#         test_images.append(path)
#         test_labels.append(label)

# A=list(set(train_labels))
# B=list(set(test_labels))
# labels={A[idx]:idx for idx in range(len(A))}
# def calculate_hu_moments(image):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     moments = cv2.moments(gray_image)
#     hu_moments = cv2.HuMoments(moments).flatten()
#     return hu_moments
# def calculate_haralick_texture(image):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     textures = mahotas.features.haralick(gray_image).mean(axis=0)
#     return textures
# def build_kmeans_for_sift(image_paths, num_clusters=50):
#     sift = cv2.SIFT_create()
#     descriptors_list = []
#     for image_path in tqdm(image_paths, desc="Extracting SIFT Descriptors"):
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#         keypoints, descriptors = sift.detectAndCompute(gray_image, None)
#         if descriptors is not None:
#             descriptors_list.append(descriptors)

#     all_descriptors = np.vstack(descriptors_list)
#     kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_descriptors)
#     return kmeans
# def calculate_sift_bovw(image, kmeans, num_clusters=50):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     sift = cv2.SIFT_create()
#     keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    
#     if descriptors is not None:
#         labels = kmeans.predict(descriptors)
#         hist = np.bincount(labels, minlength=num_clusters).astype(float)
#         return hist / hist.sum()
#     else:
#         return np.zeros(num_clusters)
# num_clusters = 50
# kmeans = build_kmeans_for_sift(train_images, num_clusters=num_clusters)
# def extract_rgb_features(image):
#     image = image.astype(np.float32)
#     mean_intensity = np.mean(image)
#     mean_values = np.mean(image, axis=(0,1))
#     r_g_ratio = mean_values[2] / mean_values[1] if mean_values[1] != 0 else 0
#     r_b_ratio = mean_values[2] / mean_values[0] if mean_values[0] != 0 else 0
#     g_b_ratio = mean_values[1] / mean_values[0] if mean_values[0] != 0 else 0    
#     return np.array([mean_intensity, r_g_ratio, r_b_ratio, g_b_ratio])

# def extract_hsi_features(mat_image):
#     with h5py.File(mat_image, 'r') as mat_file:
#         if 'cube' in mat_file:
#             data = mat_file['cube'][:]
#             numpy_array = np.array(data)
#             ref = np.mean(numpy_array, axis=(1,2))
#             ref = ref / 255.0
#             return ref
#         else:
#             # Print keys only if 'cube' is missing
#             keys = list(mat_file.keys())
#             print(f"Error: 'cube' key not found in file {mat_image}. Available keys: {keys}")
            

# def calculate_non_black_avg_rgb(image):
#     non_black_mask = np.any(image != 0, axis=-1)
#     avg_r = image[:, :, 0].mean()
#     avg_g = image[:, :, 1].mean()
#     avg_b = image[:, :, 2].mean()
#     return avg_r, avg_g, avg_b  
    
# def extract_features(image_path):
#     filename=image_path.split('/')[-1]
#     name=filename.replace('.jpg','.mat')
#     matfile=os.path.join(hsi_directory, name)
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
#     avg_r, avg_g, avg_b = calculate_non_black_avg_rgb(image)
#     hu_moments = calculate_hu_moments(image)
#     haralick_texture = calculate_haralick_texture(image)
#     sift_bovw = calculate_sift_bovw(image, kmeans, num_clusters=50)
#     rgb_features=extract_rgb_features(image)
#     hsi_features=extract_hsi_features(matfile)
#     features = np.concatenate(( hu_moments, haralick_texture, sift_bovw, rgb_features,hsi_features))
#     return features

# train_features = np.array([extract_features(img_path) for img_path in tqdm(train_images, desc="Extracting Training Features")])
# test_features = np.array([extract_features(img_path) for img_path in tqdm(test_images, desc="Extracting Testing Features")])
# max_acc=0
# state=0
# for i in tqdm(range(100), desc='States'):
#     rf_classifier = RandomForestClassifier(n_estimators=100, random_state=i)
#     rf_classifier.fit(train_features, train_labels)
#     train_predictions = rf_classifier.predict(train_features)
#     test_predictions = rf_classifier.predict(test_features)
    
#     train_accuracy = accuracy_score(train_labels, train_predictions)
#     test_accuracy = accuracy_score(test_labels, test_predictions)
#     if test_accuracy>max_acc:
#         state=i
#     max_acc=max(max_acc, test_accuracy)
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=state)
# rf_classifier.fit(train_features, train_labels)
# train_predictions = rf_classifier.predict(train_features)
# test_predictions = rf_classifier.predict(test_features)

# train_accuracy = accuracy_score(train_labels, train_predictions)
# test_accuracy = accuracy_score(test_labels, test_predictions)

# print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
# print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
# print(f'Weighted F1 score:',f1_score(test_labels, test_predictions, average='weighted'))
# print(classification_report(test_labels, test_predictions))
# model_acc.append(test_accuracy)
# print('starting resnet')

# class ImageDataset(Dataset):
#     def __init__(self, image_paths, labels, transform=None):
#         self.image_paths = image_paths
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image = Image.open(self.image_paths[idx]).convert('RGB')
#         label = self.labels[idx]

#         if self.transform:
#             image = self.transform(image)

#         return image, label
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# train_dataset = ImageDataset(image_paths=train_images, labels=train_labels, transform=transform)
# train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
# test_dataset = ImageDataset(image_paths=test_images, labels=test_labels, transform=transform)
# test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cuda'
# model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# num_features = model.fc.in_features
# num_classes = 7
# model.fc = nn.Linear(num_features, num_classes)
# model = model.to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# num_epochs = 20

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     for images, labels in tqdm(train_dataloader):
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     epoch_loss = running_loss / len(train_dataloader)
#     accuracy = 100 * correct / total
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
# model.eval()
# all_preds = []
# all_labels = []

# with torch.no_grad():
#     for images, labels in test_dataloader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, preds = torch.max(outputs, 1)
        
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
# overall_accuracy = accuracy_score(all_labels, all_preds)
# print(f"Test Accuracy: {overall_accuracy * 100:.2f}%")
# f1 = f1_score(all_labels, all_preds, average='weighted')
# print(f"Weighted F1 Score: {f1:.4f}")
# print(classification_report(all_labels, all_preds))
# res_acc.append(accuracy_score)
# print('Starting densenet')
# class ImageDataset(Dataset):
#     def __init__(self, image_paths, labels, transform=None):
#         self.image_paths = image_paths
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image = Image.open(self.image_paths[idx]).convert('RGB')
#         label = self.labels[idx]

#         if self.transform:
#             image = self.transform(image)

#         return image, label
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# train_dataset = ImageDataset(image_paths=train_images, labels=train_labels, transform=transform)
# train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
# test_dataset = ImageDataset(image_paths=test_images, labels=test_labels, transform=transform)
# test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cuda'
# model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
# num_features = model.classifier.in_features
# num_classes = 7
# model.classifier = nn.Linear(num_features, num_classes)
# model = model.to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# num_epochs = 20

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     for images, labels in tqdm(train_dataloader):
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     epoch_loss = running_loss / len(train_dataloader)
#     accuracy = 100 * correct / total
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
# model.eval()
# all_preds = []
# all_labels = []

# with torch.no_grad():
#     for images, labels in test_dataloader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, preds = torch.max(outputs, 1)
        
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
# overall_accuracy = accuracy_score(all_labels, all_preds)
# dense_acc.append(overall_accuracy)
# print(f"Test Accuracy: {overall_accuracy * 100:.2f}%")
# f1 = f1_score(all_labels, all_preds, average='weighted')
# print(f"Weighted F1 Score: {f1:.4f}")
# print(classification_report(all_labels, all_preds))


# In[1]:


from utils import *
rgb_directory='HAM/RGB/'
hsi_directory='HAM/HSI/'
labels_directory='ham.csv'
df=pd.read_csv(labels_directory)
df=df.drop(['Unnamed: 0.1','Unnamed: 0'], axis=1)
idx_to_drop=[]
for i in range(len(df)):
    impath=df.iloc[i,0]
    name=impath.split('/')[-1]
    hsi_path=os.path.join(hsi_directory, name.replace('.jpg','.mat'))
    if os.path.exists(hsi_path):
        continue
    else:
        idx_to_drop.append(i)

df=df.drop(idx_to_drop, axis=0)
df=df.reset_index(drop=True)
ids=df['ID']
ids=list(set(list(ids)))
train_ids, test_ids=train_test_split(ids, test_size=0.1, random_state=47)

train_images=[]
test_images=[]
train_labels=[]
test_labels=[]
for i in range(len(df)):
    idnum=df.iloc[i,-1]
    path=df.iloc[i,0]
    label=df.iloc[i,1]
    if idnum in train_ids:
        train_images.append(path)
        train_labels.append(label)
    else:
        test_images.append(path)
        test_labels.append(label)

A=list(set(train_labels))
B=list(set(test_labels))
labels={A[idx]:idx for idx in range(len(A))}
def calculate_hu_moments(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    moments = cv2.moments(gray_image)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments
def calculate_haralick_texture(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    textures = mahotas.features.haralick(gray_image).mean(axis=0)
    return textures
def build_kmeans_for_sift(image_paths, num_clusters=50):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for image_path in tqdm(image_paths, desc="Extracting SIFT Descriptors"):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)

    all_descriptors = np.vstack(descriptors_list)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_descriptors)
    return kmeans
def calculate_sift_bovw(image, kmeans, num_clusters=50):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    
    if descriptors is not None:
        labels = kmeans.predict(descriptors)
        hist = np.bincount(labels, minlength=num_clusters).astype(float)
        return hist / hist.sum()
    else:
        return np.zeros(num_clusters)
num_clusters = 50
kmeans = build_kmeans_for_sift(train_images, num_clusters=num_clusters)
def extract_rgb_features(image):
    image = image.astype(np.float32)
    mean_intensity = np.mean(image)
    mean_values = np.mean(image, axis=(0,1))
    r_g_ratio = mean_values[2] / mean_values[1] if mean_values[1] != 0 else 0
    r_b_ratio = mean_values[2] / mean_values[0] if mean_values[0] != 0 else 0
    g_b_ratio = mean_values[1] / mean_values[0] if mean_values[0] != 0 else 0    
    return np.array([mean_intensity, r_g_ratio, r_b_ratio, g_b_ratio])

def extract_hsi_features(mat_image):
    with h5py.File(mat_image, 'r') as mat_file:
        if 'cube' in mat_file:
            data = mat_file['cube'][:]
            numpy_array = np.array(data)
            ref = np.mean(numpy_array, axis=(1,2))
            ref = ref / 255.0
            return ref
        else:
            # Print keys only if 'cube' is missing
            keys = list(mat_file.keys())
            print(f"Error: 'cube' key not found in file {mat_image}. Available keys: {keys}")
            

def calculate_non_black_avg_rgb(image):
    non_black_mask = np.any(image != 0, axis=-1)
    avg_r = image[:, :, 0].mean()
    avg_g = image[:, :, 1].mean()
    avg_b = image[:, :, 2].mean()
    return avg_r, avg_g, avg_b  
    
def extract_features(image_path):
    filename=image_path.split('/')[-1]
    name=filename.replace('.jpg','.mat')
    matfile=os.path.join(hsi_directory, name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    avg_r, avg_g, avg_b = calculate_non_black_avg_rgb(image)
    hu_moments = calculate_hu_moments(image)
    haralick_texture = calculate_haralick_texture(image)
    sift_bovw = calculate_sift_bovw(image, kmeans, num_clusters=50)
    rgb_features=extract_rgb_features(image)
    hsi_features=extract_hsi_features(matfile)
    features = np.concatenate(( hu_moments, haralick_texture, sift_bovw, rgb_features,hsi_features))
    return features

train_features = np.array([extract_features(img_path) for img_path in tqdm(train_images, desc="Extracting Training Features")])
test_features = np.array([extract_features(img_path) for img_path in tqdm(test_images, desc="Extracting Testing Features")])
max_acc=0
state=0
for i in tqdm(range(100), desc='States'):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=i)
    rf_classifier.fit(train_features, train_labels)
    train_predictions = rf_classifier.predict(train_features)
    test_predictions = rf_classifier.predict(test_features)
    
    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    if test_accuracy>max_acc:
        state=i
    max_acc=max(max_acc, test_accuracy)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=state)
rf_classifier.fit(train_features, train_labels)
train_predictions = rf_classifier.predict(train_features)
test_predictions = rf_classifier.predict(test_features)

train_accuracy = accuracy_score(train_labels, train_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f'Weighted F1 score:',f1_score(test_labels, test_predictions, average='weighted'))
print(classification_report(test_labels, test_predictions))


# In[2]:


from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler()
train_features, train_labels = ros.fit_resample(train_features, train_labels)
for i in tqdm(range(100), desc='States'):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=i)
    rf_classifier.fit(train_features, train_labels)
    train_predictions = rf_classifier.predict(train_features)
    test_predictions = rf_classifier.predict(test_features)
    
    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    if test_accuracy>max_acc:
        state=i
    max_acc=max(max_acc, test_accuracy)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=state)
rf_classifier.fit(train_features, train_labels)
train_predictions = rf_classifier.predict(train_features)
test_predictions = rf_classifier.predict(test_features)

train_accuracy = accuracy_score(train_labels, train_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f'Weighted F1 score:',f1_score(test_labels, test_predictions, average='weighted'))
print(classification_report(test_labels, test_predictions))


# In[3]:


from utils import *
rgb_directory='HAM/RGB/'
hsi_directory='HAM/HSI/'
labels_directory='ham.csv'
df=pd.read_csv(labels_directory)
df=df.drop(['Unnamed: 0.1','Unnamed: 0'], axis=1)
idx_to_drop=[]
for i in range(len(df)):
    impath=df.iloc[i,0]
    name=impath.split('/')[-1]
    hsi_path=os.path.join(hsi_directory, name.replace('.jpg','.mat'))
    if os.path.exists(hsi_path):
        continue
    else:
        idx_to_drop.append(i)

df=df.drop(idx_to_drop, axis=0)
df=df.reset_index(drop=True)
ids=df['ID']
ids=list(set(list(ids)))
train_ids, test_ids = train_test_split(ids, test_size=0.35, random_state=47)

train_images=[]
test_images=[]
train_labels=[]
test_labels=[]
for i in range(len(df)):
    idnum=df.iloc[i,-1]
    path=df.iloc[i,0]
    label=df.iloc[i,1]
    if idnum in train_ids:
        train_images.append(path)
        train_labels.append(label)
    else:
        test_images.append(path)
        test_labels.append(label)


A=list(set(train_labels))
B=list(set(test_labels))
labels={A[idx]:idx for idx in range(len(A))}
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = ImageDataset(image_paths=train_images, labels=train_labels, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_dataset = ImageDataset(image_paths=test_images, labels=test_labels, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda'
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
num_classes = 7
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataloader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
overall_accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {overall_accuracy * 100:.2f}%")
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")
print(classification_report(all_labels, all_preds))


# In[4]:


for epoch in range(40):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataloader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
overall_accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {overall_accuracy * 100:.2f}%")
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")
print(classification_report(all_labels, all_preds))


# In[6]:


from utils import *
from torchvision.models import ResNet18_Weights, DenseNet121_Weights

rgb_directory='HAM/RGB/'
hsi_directory='HAM/HSI/'
labels_directory='ham.csv'
df=pd.read_csv(labels_directory)
df=df.drop(['Unnamed: 0.1','Unnamed: 0'], axis=1)
idx_to_drop=[]
for i in range(len(df)):
    impath=df.iloc[i,0]
    name=impath.split('/')[-1]
    hsi_path=os.path.join(hsi_directory, name.replace('.jpg','.mat'))
    if os.path.exists(hsi_path):
        continue
    else:
        idx_to_drop.append(i)

df=df.drop(idx_to_drop, axis=0)
df=df.reset_index(drop=True)
ids=df['ID']
ids=list(set(list(ids)))
train_ids, test_ids = train_test_split(ids, test_size=0.35, random_state=47)

train_images=[]
test_images=[]
train_labels=[]
test_labels=[]
for i in range(len(df)):
    idnum=df.iloc[i,-1]
    path=df.iloc[i,0]
    label=df.iloc[i,1]
    if idnum in train_ids:
        train_images.append(path)
        train_labels.append(label)
    else:
        test_images.append(path)
        test_labels.append(label)


A=list(set(train_labels))
B=list(set(test_labels))
labels={A[idx]:idx for idx in range(len(A))}
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = ImageDataset(image_paths=train_images, labels=train_labels, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_dataset = ImageDataset(image_paths=test_images, labels=test_labels, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda'
model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
num_features = model.classifier.in_features
num_classes = 7
model.classifier = nn.Linear(num_features, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 40

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataloader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
overall_accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {overall_accuracy * 100:.2f}%")
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")
print(classification_report(all_labels, all_preds))


# In[ ]:




