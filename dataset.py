import torch


class dataset_xmls(data.Dataset):
	def __init__(self, images_dir,annotation_dir,image_size,transform=None):
		self.images_dir = images_dir
		self.annotation_dir=annotation_dir
		

		assert isinstance(image_size,int) or len(image_size)==2

		if isinstance(image_size,int):
			self.H=image_size
			self.W=image_size
		else:
			self.W,self.H=image_size

		assert self.H> 0 and self.W >0

		self.list_IDs = os.listdir(self.images_dir)
		self.label_dict={}
		
	@staticmethod
	def read_xmls(xml_file):
		tree = ET.parse(xml_file)            
		root = tree.getroot()
		for member in root.findall('object'):
			value = (root.find('filename').text,
					int(math.ceil(float(root.find('size')[0].text))),
					int(math.ceil(float(root.find('size')[1].text))),
					member.find('name').text,
					int(math.ceil(float(member.find('bndbox')[0].text))),
					int(math.ceil(float(member.find('bndbox')[1].text))),
					int(math.ceil(float(member.find('bndbox')[2].text))),
					int(math.ceil(float(member.find('bndbox')[3].text)))
					)
			yield value 

	def create_annotation_dict(self,anno_path):
		boxes=[]
		labels=[]
		area=[]
		iscrowd=[]
		
		for filename, width, height, label, xmin, ymin, xmax, ymax in read_xml(xml_file):
			x0=xmin*(self.W/width)
			y0=ymin*(self.H/height)
			x1=xmax*(self.W/width)
			y1=ymax*(self.H/height)


			assert 0<= y0 <=self.H and 0<= y1 <=self.H
			assert 0<= x0 <=self.W and 0<= x1 <=self.W
			assert label in self.slabel_dict

			boxes.append([x0, y0, x1, y1])
			labels.append(self.label_dict[label])
			area.append((y1-y0)*(x1-x0))
			iscrowd.append(False)

		anno_dict={'boxes':torch.FloatTensor(boxes),
			'labels':torch.Int64Tensor(labels),
			'area':torch.FloatTensor(area),
			'iscrowd':torch.UInt8Tensor(iscrowd),
		}

		return anno_dict

	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		image_id=index
		image_name = self.list_IDs[image_id]

		image_path=os.path.join(self.images_dir,image_name)
		anno_path=os.path.join(self.annotation_dir,image_name.replace('.jpg','xml'))
		
		assert os.path.exists(image_path) and os.path.exists(anno_path) 

		X=Image.open(image_path)
		X=X.resize((self.W,self.H))

		anno_dict=self.create_annotation_dict(anno_path)
		anno_dict['image_id']=torch.Int64Tensor(image_id)

		if self.transform:
			X= self.transform(X)


		return X,anno_dict
				