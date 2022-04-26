# DeepLearningProject

# 3D reconstruction of objects from a single 2D RGB image is considered as a core problem in the field of computer vision. Many tasks like navigation in real-time systems such as autonomous vehicles directly depend on this problem. Whereas, there are several good proposed solutions for this problem, still accuracy and computational time are required to be enhanced for real-time and delicate application. In this work, we want to tackle this problem by proposing several network architectures in a supervised pipeline to reconstruct the 3D point cloud geometry of an object depicted in the input image. We designed a network as the base model that tries to find the appropriate position of each point in the output point cloud, based on pixel of 2D input image. After analysis of initial model performance in terms of computational time and accuracy, we built two other models based on the concepts of Self-Attention \cite{attention} and Vision Transformers \cite{Vit} then they are analyzed as the first model. All the models are evaluated by the Shapenet dataset \cite{Shapenet} and our results compared to the Point Set Generation Network (PSGN) \cite{PSGN} which is a well-known and baseline contributions in this research area. Furthermore, We have depicted our models are able to outperform PSGN in terms of time-efficiency and accuracy.
