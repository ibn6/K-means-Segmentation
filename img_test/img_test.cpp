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

struct BGR_Centroid;

/// Un pixel del cluster
struct BGR_Elem {
	/// Pixel de la imagen
	Pixel* pix;
	/// Distancia al centro del cluster
	int dist;
	int row, col;

	BGR_Centroid* centroid;

	BGR_Elem(Pixel* pix = NULL, const int& row = -1, const int& col = -1, BGR_Centroid* centroid = NULL, const int& dist = -1) {
		this->pix = pix;
		this->row = row;
		this->col = col;
		this->centroid = centroid;
		this->dist = dist;
	}

	~BGR_Elem() {
		centroid = NULL;
		pix = NULL;
	}
};

/// Centro del cluster
struct BGR_Centroid {
	/// Color del centro
	Pixel pix;
	int row, col;

	BGR_Elem elem;

	/// Pixels cuyo centro es este centro.
	std::vector<BGR_Elem> cluster;

	BGR_Centroid(Pixel& pix, const int& row = -1, const int& col = -1, BGR_Elem* elem = NULL) {
		this->pix = pix;
		this->row = row;
		this->col = col;
		if (elem != NULL)
			this->elem = *elem;
	}
	BGR_Centroid() {

	}
};

void printCentroid(const BGR_Centroid& c1, const int& i, const int& k, const double& dist) {
	std::cout << "(" << i << "/" << k << ")" << "\tPix: [" << (int)c1.pix.x << ", " << (int)c1.pix.y << ", " << (int)c1.pix.z << "]" << std::setw(16)
		<< "\tRow: " << c1.row << std::setw(8) << "\tCol: " << c1.col << std::setw(8) << "\tDist: " << dist << "." << std::endl;
}

void printCluster(const BGR_Centroid& c1, const size_t& x, const size_t& y, const size_t& z) {
	std::cout << "\tCurrent color: [" << (int)c1.pix.x << ", " << (int)c1.pix.y << ", " << (int)c1.pix.z << "]\t" //<< std::endl
		<< std::setw(30) << "Old color: [" << x << ", " << y << ", " << z << "]";
}

/// k-means++
// No implementar en asm, se puede sustituir por un generador de numeros aleatorios.
std::vector<BGR_Centroid> BGR_centroids(cv::Mat& img, const int& k) {
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
					int dist = distance(centroid.pix, *elem.pix);
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
				newCentroid.pix = *newCentroid.elem.pix;
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

//TODO implementar esta funcion en asm
void segment(cv::Mat& img, std::vector<BGR_Centroid>& centroids) {
	time_t timer = clock();
	bool modified;
	int i = 0;
	do {
		++i;
		std::cout << "Iteration " << i << ":\n";
		modified = false;
		// TODO Solo implementar en asm este doble bucle for

		/// Organizar cada pixel. Un cluster se forma por los pixeles mas cercanos a su centro.
		for (int i = 0; i < img.rows; ++i) {
			for (int j = 0; j < img.cols; ++j) {
				BGR_Elem elem(&img.at<Pixel>(i, j), i, j);
				for (auto& centroid : centroids) {
					int dist = distance(centroid.pix, *elem.pix);
					if (elem.centroid == NULL || dist < elem.dist) {
						elem.dist = dist;
						elem.centroid = &centroid;
					}
				}
				elem.centroid->cluster.push_back(elem);
			}
		}
 		// A partir de aqui no si se ve que es complicado.
		// Si es facil creo lo podemos hacer tambien.

		/// Se saca el nuevo color de cada cluster. Si el nuevo color de algun cluster es distinto al color de su centro, se vuelven a organizar los pixeles. 
		/// Si no, todos los pixeles ya estan organizados.
		for (auto& centroid : centroids) {
			size_t x = 0, y = 0, z = 0;
			for (auto& elem : centroid.cluster) {
				x += elem.pix->x;
				y += elem.pix->y;
				z += elem.pix->z;
			}
			x /= centroid.cluster.size();
			y /= centroid.cluster.size();
			z /= centroid.cluster.size();
			printCluster(centroid, x, y, z);
			if (x != centroid.pix.x || y != centroid.pix.y || z != centroid.pix.z) {
				//El nuevo color es distinto del anterior.
				Pixel aux(x, y, z);
				std::cout << "\tDifference: " << distance(aux, centroid.pix);
				centroid.pix = aux;
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
			*elem.pix = centroid.pix;

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
	std::string file = argv[1];
	int k = std::stoi(argv[2]);
	return BGR_segmentation(file, k);
}

