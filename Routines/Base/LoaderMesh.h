#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"
#include <map>

struct Mesh {
	Eigen::Matrix<float, -1, -1> V;
	Eigen::Matrix<uint32_t, -1, -1> F;
};

inline static void load_mesh(std::string filename, Mesh &mesh) {
	if (filename.find(".obj") != std::string::npos) { 
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string err;
		bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename.c_str());
		std::cout << err << std::endl;
		assert(ret);

		std::vector<int> Ftmp;

		// Loop over shapes
		for (size_t s = 0; s < shapes.size(); s++) {
			// Loop over faces(polygon)
			size_t index_offset = 0;
			for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
				int fv = shapes[s].mesh.num_face_vertices[f];

				// Loop over vertices in the face.
				for (size_t v = 0; v < (size_t)fv; v++) {
					// access to vertex
					Ftmp.push_back(shapes[s].mesh.indices[index_offset + v].vertex_index);
				}
				index_offset += fv;
			}	
		}

		mesh.F.resize(3, Ftmp.size()/3);	
		std::copy(Ftmp.begin(), Ftmp.end(), mesh.F.data());
		mesh.V.resize(3, attrib.vertices.size()/3);	
		for (int i = 0; i < (int)attrib.vertices.size(); i++)
			mesh.V(i) = attrib.vertices[i];

		printf("file: %s\n", filename.c_str());
		printf("n-faces: %d\n", (int)mesh.F.cols());
		printf("n-verts: %d\n", (int)mesh.V.cols());

	} else if (filename.find(".ply") != std::string::npos) { 
		std::ifstream ss(filename);
		tinyply::PlyFile file;
		file.parse_header(ss);
		std::shared_ptr<tinyply::PlyData> vertices, elements;

		try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
		catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

		try { elements = file.request_properties_from_element("face", { "vertex_indices" }, 3); }
		catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

		file.read(ss);

		const size_t n_vertices_bytes = vertices->buffer.size_bytes();
		mesh.V.resize(3, vertices->count);
		std::memcpy(mesh.V.data(), vertices->buffer.get(), n_vertices_bytes);
		
		const size_t n_elements_bytes = elements->buffer.size_bytes();
		mesh.F.resize(3, elements->count);
		std::memcpy(mesh.F.data(), elements->buffer.get(), n_elements_bytes);
		
		printf("file: %s\n", filename.c_str());
		printf("n-faces: %d\n", (int)mesh.F.cols());
		printf("n-verts: %d\n", (int)mesh.V.cols());
	} else {
		fprintf(stderr, "Error: Mesh format not known.\n");
		exit(1);
	}
	assert(mesh.V.cols() > 0 && mesh.F.cols() > 0 && "Error loading mesh.");
}

