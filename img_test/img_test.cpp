//TODO implementar funcion segment(...) en asm.

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui.hpp> 

#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <iomanip>

typedef cv::Point3_<uint8_t> Pixel;
typedef cv::Point3_<float> Pixelf;

/// Distancia euclídea
int distance(const Pixel& p1, const Pixel& p2) {
	return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2) + std::pow(p1.z - p2.z, 2));
}

int distance_asm(const int& p1x, const int& p1y, const int& p1z, const int& p2x, const int& p2y, const int& p2z) {
	return std::sqrt(std::pow(p1x - p2x, 2) + std::pow(p1y - p2y, 2) + std::pow(p1z - p2z, 2));
}

struct BGR_Centroid;

/// Un pixel del cluster
struct BGR_Elem {
	/// Pixel de la imagen
	Pixel* img_pixel;
	/// Distancia al centro del cluster
	int dist;
	int row, col;

	BGR_Centroid* centroid;

	BGR_Elem(Pixel* pix = NULL, const int& row = -1, const int& col = -1, BGR_Centroid* centroid = NULL, const int& dist = -1) {
		this->img_pixel = pix;
		this->row = row;
		this->col = col;
		this->centroid = centroid;
		this->dist = dist;
	}

	~BGR_Elem() {
		centroid = NULL;
		img_pixel = NULL;
	}
};

/// Centro del cluster
struct BGR_Centroid {
	/// Color del centro
	Pixel BGR_color;
	int row, col;

	BGR_Elem elem;

	/// Pixels cuyo centro es este centro.
	std::vector<BGR_Elem> cluster;

	BGR_Centroid(Pixel& pix, const int& row = -1, const int& col = -1, BGR_Elem* elem = NULL) {
		this->BGR_color = pix;
		this->row = row;
		this->col = col;
		if (elem != NULL)
			this->elem = *elem;
	}
	BGR_Centroid() {

	}
};

void printCentroid(const BGR_Centroid& c1, const int& i, const int& k, const double& dist) {
	std::cout << "(" << i << "/" << k << ")" << "\tPix: [" << (int)c1.BGR_color.x << ", " << (int)c1.BGR_color.y << ", " << (int)c1.BGR_color.z << "]" << std::setw(16)
		<< "\tRow: " << c1.row << std::setw(8) << "\tCol: " << c1.col << std::setw(8) << "\tDist: " << dist << "." << std::endl;
}

void printCluster(const BGR_Centroid& c1, const size_t& x, const size_t& y, const size_t& z) {
	std::cout << "\tCurrent color: [" << (int)c1.BGR_color.x << ", " << (int)c1.BGR_color.y << ", " << (int)c1.BGR_color.z << "]\t" //<< std::endl
		<< std::setw(30) << "Old color: [" << x << ", " << y << ", " << z << "]";
}

/// k-means++
// No implementar en asm, se puede sustituir por un generador de numeros aleatorios.
std::vector<BGR_Centroid> BGR_centroids(cv::Mat& img, const unsigned int& k) {
	clock_t timer = clock();
	int col = rand() % img.cols;
	int row = rand() % img.rows;

	BGR_Centroid randomCentroid(img.at<Pixel>(row, col), row, col);

	std::vector<BGR_Centroid> centroids;
	centroids.push_back(randomCentroid);
	BGR_Elem trash;
	int counter = 1;
	printCentroid(randomCentroid, counter, k, 0);

	while (centroids.size() < k) {
		counter++;
		for (int i = 0; i < img.rows; ++i) {
			for (int j = 0; j < img.cols; ++j) {
				BGR_Elem elem(&img.at<Pixel>(i, j), i, j);
				bool cont = false;

				for (auto& centroid : centroids) {
					if (centroid.col == elem.col && centroid.row == elem.row) {
						cont = true;
						break;
					}
					int dist = distance(centroid.BGR_color, *elem.img_pixel);
					if (elem.dist == -1 || dist < elem.dist) {
						elem.dist = dist;
						elem.centroid = &centroid;
					}
				}
				if (cont)
					continue;

				if (elem.dist > elem.centroid->elem.dist)
					elem.centroid->elem = elem;
			}
		}
		BGR_Centroid newCentroid;
		bool first = true;
		double dist = 0;
		for (auto& centroid : centroids) {
			if (first || centroid.elem.dist > newCentroid.elem.dist) {
				dist = centroid.elem.dist;
				newCentroid.elem = centroid.elem;
				newCentroid.col = newCentroid.elem.col;
				newCentroid.row = newCentroid.elem.row;
				newCentroid.BGR_color = *newCentroid.elem.img_pixel;
				first = false;
			}
		}
		centroids.push_back(newCentroid);
		for (auto& centroid : centroids)
			centroid.elem = trash;
		printCentroid(newCentroid, counter, k, dist);
	}
	std::cout << (double)(clock() - timer) / CLOCKS_PER_SEC << " seconds." << std::endl;
	return centroids;
}

/// Funcion auxiliar para llamar al metodo push_back()
void push_back(std::vector<BGR_Elem>& vec, BGR_Elem& elem) {
	vec.push_back(elem);
}

/// Convertir matriz a std::vector
/// Para omitir el metodo img.at<>();
std::vector<Pixel> mat2vec(cv::Mat& mat) {
	std::vector<Pixel> array;
	for (int i = 0; i < mat.rows; ++i)
		for (int j = 0; j < mat.cols; ++j)
			array.push_back(mat.at<Pixel>(i, j));
	/// Se accede a un pixel con array[i]
	/// Se accede a los colores con array[i].x, array[i].y, array[i].z

	return array;
}


void organize_asm(cv::Mat& img, std::vector<BGR_Centroid>& centroids) {
	std::cout << "asdasda: " << sizeof(centroids[0].BGR_color) << std::endl;
	std::cout << "dddddd: " << sizeof(uint8_t) << std::endl;
	BGR_Elem elem;
	Pixel aux_pixel;
	Pixel aux_pixel2;
	uint8_t aux_num;
	Pixel* ptr = &aux_pixel;
	int tam_array_centroids = centroids.size();
	auto aaa = mat2vec(img);
	Pixel* img_array = aaa.data(); //Llamamos al metodo que te tranforma img en un array
	std::cout << "dddddd: " << sizeof(img_array[0]) << std::endl;
	BGR_Centroid* centroids_array = centroids.data();
	auto aa = &centroids[0];

	int tam_pixel = sizeof(Pixel);
	int tam_centroid = sizeof(BGR_Centroid);
	int rows = img.rows;
	int cols = img.cols;
	__asm {
		mov ebx, 0 //Contador exterior de rows i
		mov ecx, 0 //Contador medio de cols j
		mov edi, 0 //Contador interior
		//mov eax, vec 
		mov esi, centroids_array // Si no se puede hacer esto yo haría un metodo que me transforme el vector en un array

		bucle_exterior :
			mov edx, rows
			cmp ebx, edx
			je fin_exterior

				bucle_medio :
				mov edx, cols
				cmp ecx, edx
				je fin_medio

			//Inicio constructor elem
				mov eax, rows
				mul ebx // rows * i
				add eax, ecx // (rows * i) + j
				mul tam_pixel // sizeof(pixel) * ((rows * i) + j)
				mov edx, img_array 
				add eax, edx // img_array + sizeof(pixel) * ((rows * i) + j) // img_array[i][j]
				mov elem.img_pixel, eax //Guardo la referencia del pixel que almacena la matriz (img)
				mov elem.row, ebx
				mov elem.col, ecx
			//final constructor elem
				push ebx //guardar fila
				push ecx //guardar columna

				mov ecx, 0 //Primera iteracion
			bucle_interior: //INICIO TERCER BUCLE  puedo usar eax ebx edx
					cmp edi, tam_array_centroids
					je fin_interior

					

			//Inicio metodo distance
					mov eax, tam_centroid
					mul edi
					add eax, esi //Estamos guardando centroids[k]
					// aux_pixel.x: Centroid.color
					lea edx, [eax]BGR_Centroid.BGR_color.x
					lea ebx, aux_num
					mov edx, [edx]
					mov [ebx], edx
					push [ebx]
					// aux_pixel.y: Centroid.color
					lea edx, [eax]BGR_Centroid.BGR_color.y
					lea ebx, aux_num
					mov edx, [edx]
					mov [ebx], edx
					push[ebx]
					// aux_pixel.z: Centroid.color
					lea edx, [eax]BGR_Centroid.BGR_color.z
					lea ebx, aux_num
					mov edx, [edx]
					mov [ebx], edx
					push[ebx]

					lea edx, aux_pixel

					//push aux_pixel //push centroids[k].color
					//mov ebx, [edx]Pixel.x
					//push [edx]Pixel.x
					//push [edx]Pixel.y
					//push [edx]Pixel.z

					// aux_pixel.x: Elem.color
					lea edx, [elem.img_pixel]Pixel.x
					lea ebx, aux_num
					mov edx, [edx]
					mov [ebx], edx
					push[ebx]
					// aux_pixel.y: Elem.color
					lea edx, [elem.img_pixel]Pixel.y
					lea ebx, aux_num
					mov edx, [edx]
					mov [ebx], edx
					push[ebx]
					// aux_pixel.z: Elem.color
					lea edx, [elem.img_pixel]Pixel.z
					lea ebx, aux_num
					mov edx, [edx]
					mov [ebx], edx
					push[ebx]

					lea edx, aux_pixel2
						push [edx]Pixel.x
						push [edx]Pixel.y
						push [edx]Pixel.z

					//mov edx, [elem.img_pixel]
					//push aux_pixel2 //push elem.color
					mov ebx, eax //Estamos guardando centroids[k]
					call distance_asm //Llamamos al metodo distacia. El resultado se guarda en eax
						// eax = distance(centroids[k].BGR_color, elem.img_pixel)
					pop edx // Limpiar la pila
					pop edx
					pop edx
						pop edx // Limpiar la pila
						pop edx
						pop edx
			//Fin metodo distance

			//Inicio if
					mov edx, 0
					cmp ecx, edx
					je saltar_cond // Primera iteracion, 'or' cortocircuitado. (Se necesita que siempre pase en la primera iteracion)
					mov edx, elem.dist
					cmp eax, edx //Comparamos si elem.dist es mayor o igual que eax, que entonces no entra en la condición
					jae fin_cond
			saltar_cond :
					mov ecx, 1
			//Final if

					mov elem.dist, eax
					mov elem.centroid, ebx //al pasar la referencia  yo creo que seria sin corchetes

			fin_cond :


		inc edi
			jmp bucle_interior
			fin_interior : //FIN TERCER BUCLE

		mov ebx, elem.centroid
		mov ebx, [ebx].cluster
		push ebx 
		mov ebx, elem 
		push ebx 
		call push_back
		pop ebx 
		pop ebx

			pop ecx
			pop ebx
			inc ecx
			jmp bucle_medio
			fin_medio :

		inc ebx
			jmp bucle_exterior
			fin_exterior :


	}

}

//TODO implementar esta funcion en asm
void segment(cv::Mat& img, std::vector<BGR_Centroid>& centroids) {
	time_t timer = clock();
	bool modified;
	int i = 0;
	// auto img_vec = mat2vec(mat);
	do {
		++i;
		std::cout << "Iteration " << i << ":\n";
		modified = false;
		// TODO Solo implementar en asm este doble bucle for

		/// Organizar cada pixel. Un cluster se forma por los pixeles mas cercanos a su centro.
		organize_asm(img, centroids);
		/*for (int i = 0; i < img.rows; ++i) {
			for (int j = 0; j < img.cols; ++j) {
				BGR_Elem elem(&img.at<Pixel>(i, j), i, j);
				for (auto& centroid : centroids) {
					int dist = distance(centroid.BGR_color, *elem.img_pixel);
					/// Si en la primera iteracion o la distancia entre el pixel y el nuevo centro es menor que en la anterior.
					if (elem.centroid == NULL || dist < elem.dist) {
						elem.dist = dist;
						elem.centroid = &centroid;
					}
				}
				/// Pixel se añade al cluster.
				elem.centroid->cluster.push_back(elem);
			}
		}*/
 		// A partir de aqui no si se ve que es complicado.
		// Si es facil creo lo podemos hacer tambien.

		/// Se saca el nuevo color de cada cluster. Si el nuevo color de algun cluster es distinto al color de su centro, se vuelven a organizar los pixeles. 
		/// Si no, todos los pixeles ya estan organizados.
		for (auto& centroid : centroids) {
			size_t x = 0, y = 0, z = 0;
			for (auto& elem : centroid.cluster) {
				x += elem.img_pixel->x;
				y += elem.img_pixel->y;
				z += elem.img_pixel->z;
			}
			x /= centroid.cluster.size();
			y /= centroid.cluster.size();
			z /= centroid.cluster.size();
			printCluster(centroid, x, y, z);
			if (x != centroid.BGR_color.x || y != centroid.BGR_color.y || z != centroid.BGR_color.z) {
				//El nuevo color es distinto del anterior.
				Pixel aux(x, y, z);
				std::cout << "\tDifference: " << distance(aux, centroid.BGR_color);
				centroid.BGR_color = aux;
				modified = true;
			}
			std::cout << std::endl;
		}
		if (modified)
			for (auto& centroid : centroids)
				centroid.cluster.clear();
	} while (modified);

	std::cout << "Done" << std::endl;

	/// Se pintan todos los pixeles de un cluster con el color de su centro.
	for (auto& centroid : centroids)
		for (auto& elem : centroid.cluster)
			*elem.img_pixel = centroid.BGR_color;

	std::cout << (double)(clock() - timer) / CLOCKS_PER_SEC << " seconds." << std::endl;
}

int BGR_segmentation(const std::string& file, const int& k) {
	auto img = cv::imread(file);
	if (!img.data) {
		std::cerr << "Error reading file." << std::endl;
		return -1;
	}

	auto centroids = BGR_centroids(img, k);
	//img.convertTo(img_hs, CV_32FC3, 1 / 255.0);
	//cv::cvtColor(img, img_hsv, CV_BGR2Lab);
	segment(img, centroids);
	//cv::cvtColor(img_hsv, img, CV_Lab2BGR);
	//img_hs.convertTo(img, 16, 255);

	cv::namedWindow(file, cv::WINDOW_NORMAL);
	cv::imshow(file, img);
	cv::imwrite("out-" + file, img);

	cv::waitKey(0);

	return 0;
}

int main(int argc, char**argv)
{
	if (argc != 3) {
		std::cerr << "Use: " << argv[0] << " <image> <k>" << std::endl;
		return -1;
	}
	
	//std::string file = argv[1];
	//int k = std::stoi(argv[2]);
	std::string file = "test2.jpg";
	int k = 3;
	return BGR_segmentation(file, k);
}

