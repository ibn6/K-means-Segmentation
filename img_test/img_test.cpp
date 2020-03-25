// Importante! (asm):
// Pixel = pix.x + pix.y << 8 + pix.z << 16

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
//typedef cv::Point3_<float> Pixelf;

/// Distancia euclídea
extern "C" int distance(const Pixel p1, const Pixel p2) {
	return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2) + std::pow(p1.z - p2.z, 2));
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
		dist = 0;
		row = 0;
		col = 0;
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

void printPixel(const Pixel pix) {
	std::cout << "Pix: [" << (int)pix.x << ", " << (int)pix.y << ", " << (int)pix.z << "]";
}

void printCentroid(const BGR_Centroid& c1, const int& i, const int& k, const double& dist) {
	std::cout << "(" << i << "/" << k << ")\t";
	printPixel(c1.BGR_color);
	std::cout << std::setw(16) << "\tRow: " << c1.row << std::setw(8) << "\tCol: " << c1.col << std::setw(8) << "\tDist: " << dist << "." << std::endl;
}

/// Prints the new and old color of a centroid
void printUpdatedCentroidColor(const BGR_Centroid& c1, const size_t& x, const size_t& y, const size_t& z) {
	std::cout << "\tCurrent color: [" << (int)c1.BGR_color.x << ", " << (int)c1.BGR_color.y << ", " << (int)c1.BGR_color.z << "]\t" //<< std::endl
		<< std::setw(30) << "New color: [" << x << ", " << y << ", " << z << "]";
}

// No implementar en asm, se puede sustituir por un generador de numeros aleatorios.
/// k-means++
/// The first centroid is a random pixel from the image, 
/// the 2nd as the color further away from the first,
/// and so on until there are k centroids.
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
		/// Organize each pixel in a cluster
		for (int i = 0; i < img.rows; ++i) {
			for (int j = 0; j < img.cols; ++j) {
				BGR_Elem elem(&img.at<Pixel>(i, j), i, j);
				bool cont = false;

				for (auto& centroid : centroids) {
					if (centroid.col == elem.col && centroid.row == elem.row) {
						/// The current pixel is already chosen as a centroid
						cont = true;
						break;
					}
					int dist = distance(centroid.BGR_color, *elem.img_pixel);
					/// To which cluster
					if (elem.dist == -1 || dist < elem.dist) {
						elem.dist = dist;
						elem.centroid = &centroid;
					}
				}
				if (cont)
					continue;

				/// Only the furthest away is needed
				if (elem.dist > elem.centroid->elem.dist)
					elem.centroid->elem = elem;
			}
		}
		BGR_Centroid newCentroid;
		bool first = true;
		double dist = 0;
		/// Look for the new centroid (the furthest away)
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
		/// Clean-up
		for (auto& centroid : centroids)
			centroid.elem = trash;
		printCentroid(newCentroid, counter, k, dist);
	}
	std::cout << "Created centroids in: ";
	std::cout << (double)(clock() - timer) / CLOCKS_PER_SEC << " seconds." << std::endl;
	return centroids;
}

/// Helper function to call the push_back() method on a vector from inside __asm blocks
extern "C" void push_back(std::vector<BGR_Elem>* vec, BGR_Elem* elem) {
	vec->push_back(*elem);
}

/// Convertir matriz a std::vector
/// Para omitir el metodo img.at<>();
std::vector<Pixel> mat2vec(cv::Mat& mat) {
	std::vector<Pixel> array;
	if (mat.isContinuous()) {
		// array.assign((float*)mat.datastart, (float*)mat.dataend); // <- has problems for sub-matrix like mat = big_mat.row(i)
		array.assign((Pixel*)mat.data, (Pixel*)mat.data + mat.total());
	}
	else {
		for (int i = 0; i < mat.rows; ++i) {
			array.insert(array.end(), mat.ptr<Pixel>(i), mat.ptr<Pixel>(i) + mat.cols);
		}
	}
	/// Se accede a un pixel con array[i]
	/// Se accede a los colores con array[i].x, array[i].y, array[i].z

	return array;
}

/// Organizes each pixel in a image to a centroid's cluster
void organize(cv::Mat& img, std::vector<BGR_Centroid>& centroids) {
	for (int i = 0; i < img.rows; ++i) {
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
	}
}

/// asm implementation of the organize function
void organize_asm(cv::Mat& img, std::vector<BGR_Centroid>& centroids) {
	uchar* img_array_uchar = img.data;
	int total = img.cols * img.rows;
	int pixel_size = sizeof(Pixel);
	BGR_Centroid* centroids_array = &centroids[0];
	int centroids_array_size = centroids.size();
	int centroid_size = sizeof(BGR_Centroid);
	BGR_Elem aux_elem;

	__asm {
		mov ebx, 0 // i
		mov esi, img_array_uchar // img
		// Doble bucle. Tratamos la imagen 2d como un array 1d por simplicidad y porque las coordenadas de cada pixel no importan.
		_img_loop :
		mov	eax, total
			cmp eax, ebx
			je _end_img_loop
			push ebx

			// Se pueden usar eax, ebx, ecx, edx
			// Constructor aux_elem. Las coordenadas no son importantes y no las pongo por simplicidad.
			mov aux_elem.img_pixel, esi
			mov ebx, 0
			mov ecx, centroids_array // centroids
			// Range-based for loop
			_centroid_loop :
		mov eax, centroids_array_size
			cmp eax, ebx
			je _end_centroid_loop

			// Se pueden usar registros eax, edx

			push ebx
			// Llamada metodo distance
			// Pixel imagen
			movzx eax, [esi] //Pixel.x
			movzx edx, [esi + 1] //Pixel.y
			shl edx, 8
			add eax, edx
			movzx edx, [esi + 2] //Pixel.z
			shl edx, 16
			add eax, edx
			push eax

			// Pixel centroid
			lea ebx, [ecx]BGR_Centroid.BGR_color
			movzx eax, [ebx]Pixel.x
			movzx edx, [ebx]Pixel.y
			shl edx, 8
			add eax, edx
			movzx edx, [ebx]Pixel.z
			shl edx, 16
			add eax, edx
			push eax
			mov ebx, ecx

			call distance

			mov ecx, ebx
			pop ebx
			pop ebx

			pop ebx
			mov edx, 0
			cmp ebx, edx
			je _or_cortocircuitado // Primera iteracion, no se comprueba la segunda parte del or
			mov edx, aux_elem.dist
			cmp eax, edx
			jae _fin_if
			_or_cortocircuitado :

		mov aux_elem.dist, eax
			mov aux_elem.centroid, ecx

			_fin_if :

		inc ebx
			mov eax, centroid_size
			add ecx, eax
			jmp _centroid_loop
			_end_centroid_loop :

		lea eax, aux_elem
			push eax
			mov eax, aux_elem.centroid
			lea eax, [eax].cluster
			push eax

			call push_back

			pop eax
			pop eax
			mov aux_elem.dist, 0

			pop ebx
			inc ebx
			mov eax, pixel_size
			add esi, eax
			jmp _img_loop
			_end_img_loop :
	}
}

// Implementar lo que queda de la funcion en ensamblador???
void segment(cv::Mat& img, std::vector<BGR_Centroid>& centroids, bool ensamblador) {
	time_t timer = clock(); 
	bool modified;
	int i = 0;
	// auto img_vec = mat2vec(mat);
	do {
		++i;
		std::cout << "Iteration " << i << ":\n";
		modified = false;

		/// Organizar cada pixel. Un cluster se forma por los pixeles mas cercanos a su centro.
		ensamblador ?
			organize_asm(img, centroids) :
			organize(img, centroids);

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
			printUpdatedCentroidColor(centroid, x, y, z);
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
	std::string asm_or_cpp = ensamblador ? "Asm: " : "C++: ";
	std::cout << asm_or_cpp;
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
	segment(img, centroids, false);
	//cv::cvtColor(img_hsv, img, CV_Lab2BGR);
	//img_hs.convertTo(img, 16, 255);

	cv::namedWindow(file, cv::WINDOW_NORMAL);
	cv::imshow(file, img);
	cv::imwrite(k + "out-" + file, img);

	cv::waitKey(0);

	return 0;
}

int main(int argc, char**argv)
{
	if (argc != 3) {
		std::cerr << "Use: " << argv[0] << " <image> <k>" << std::endl;
		return -1;
	}

	std::string file = argv[1];
	int k = std::stoi(argv[2]);
	//std::string file = "test.jpg";
	//int k = 3;
	return BGR_segmentation(file, k);
}

