/*
 * map_widget.cpp
 *
 *  Created on: Dec 10, 2017
 *      Author: alex
 */

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <memory>
#include <cstdlib>


using namespace std;

template<typename T>
struct coord2D_t { T x, y; };

template<typename T>
struct rect2D_t  { T x, y, size_x, size_y; };

// Returns true if two rectangles (l1, r1) and (l2, r2) overlap
template<typename T>
char rect_overlap(const struct rect2D_t<T>& r1, const struct rect2D_t<T>& r2)
{
if (r1.size_x < r2.x || r2.size_x < r1.x) return false; // If one rectangle is on left side of other
if (r1.size_y < r2.y || r2.size_y < r1.y) return false; // If one rectangle is above of other
return true;
}

//This object makes a photo marker of a player
class MapMarkerWidget{
public:
	enum ORIENT : unsigned int { ORIENT_NW, ORIENT_NE, ORIENT_SW, ORIENT_SE, ORIENT_LAST };
    unsigned int default_orient;
    const unsigned int& drawn_orient; //It equals -1 if the marker was not drawn
    const rect2D_t<int>& drawn_frame;
private:
    unsigned int _drawn_orient;
	coord2D_t<unsigned int> border;   //Container (photo) board
	coord2D_t<unsigned int> length;   //Container (photo) holder
	coord2D_t<unsigned int> offset;   //Container (photo) offset for NW shape
	coord2D_t<float       > fshape;   //Container fraction shape
	coord2D_t<unsigned int> origin;   //Container origin
	unsigned char color[3];  //Container color
	cv::Mat src_pimage, pimage, fimage_mask[ORIENT::ORIENT_LAST], back_pimage, back_fimage; //Images
    rect2D_t<int> _drawn_frame, frame[ORIENT::ORIENT_LAST]; //The rectangle container

	//This function change widget orientation as it is required by image borders; when in doubt it uses cw orientation rule
	unsigned int forced_drawn_orient(cv::Mat& background_image) {
		coord2D_t<unsigned int> size = { .x = this->offset.x + this->length.x, .y = this->offset.y + this->length.y };
		if (this->origin.x >= size.x) {
			if ((int)(this->origin.x + size.x) <= background_image.cols) {
            	if (this->origin.y >= size.y) {
        			if ((int)(this->origin.y + size.y) <= background_image.rows) return (unsigned int)-1; //No limitations
                	else return ( (this->default_orient == ORIENT::ORIENT_NE) || (this->default_orient == ORIENT::ORIENT_NW) ) ? (unsigned int)-1 : ORIENT::ORIENT_NW; //Bottom-most side
            	}
            	else return ( (this->default_orient == ORIENT::ORIENT_SE) || (this->default_orient == ORIENT::ORIENT_SW) ) ? (unsigned int)-1 : ORIENT::ORIENT_SE; //Top-most side
			}
			else {//Right-most size
            	if (this->origin.y >= size.y) {
        			if ((int)(this->origin.y + size.y) <= background_image.rows) return (this->default_orient == ORIENT::ORIENT_SE) || (this->default_orient == ORIENT::ORIENT_NE) ? (unsigned int)-1 : ORIENT::ORIENT_SE; //Right-most side
        			else return (this->default_orient == ORIENT::ORIENT_NE) ? (unsigned int)-1 : ORIENT::ORIENT_NE; // Right-bottom corner
            	}
            	else return (this->default_orient == ORIENT::ORIENT_SE) ? (unsigned int)-1 : ORIENT::ORIENT_SE; //Right-top corner
			}
		}
	    else { //Left-most size
        	if (this->origin.y >= size.y) {
    			if ((int)(this->origin.y + size.y) <= background_image.rows) return (this->default_orient == ORIENT::ORIENT_SW) || (this->default_orient == ORIENT::ORIENT_NW) ? (unsigned int)-1 : ORIENT::ORIENT_NW; //Left-most side
    			else return (this->default_orient == ORIENT::ORIENT_NW) ? (unsigned int)-1 : ORIENT::ORIENT_NW; // Left-bottom corner
        	}
        	else return (this->default_orient == ORIENT::ORIENT_SW) ? (unsigned int)-1 : ORIENT::ORIENT_SW; //Left-top corner
		}
	}

public:
	//This function defines container geometry
	MapMarkerWidget(unsigned char color[3], float fshape_x, float fshape_y, unsigned int border_x, unsigned int border_y, unsigned int length_x, unsigned int length_y, const cv::Mat& src_pimage) :
		drawn_orient(this->_drawn_orient),
        drawn_frame(this->_drawn_frame) {
        if ( (fshape_x < 0.)||(fshape_x > 1.)||(fshape_y < 0.)||(fshape_y > 1.)||(!length_x)||(!length_y) ) throw "failure in PhWidget features";
		this->fshape.x = fshape_x, this->fshape.y = fshape_y, this->border.x = border_x, this->border.y = border_y, this->length.x = length_x + 2*border_x, this->length.y = length_y + 2*border_y;
		this->offset.x=(unsigned int)std::round((float)this->length.x*this->fshape.x), this->offset.y=(unsigned int)std::round((float)this->length.y*this->fshape.y);
		this->default_orient = ORIENT::ORIENT_NW, this->origin.x = 0, this->origin.y = this->offset.y + this->length.y, this->src_pimage = src_pimage;
        //Set integer geometry
        this->color[0] = color[0], this->color[1] = color[1], this->color[2] = color[2];
        this->back_pimage = cv::Mat(this->length.y,                this->length.x, CV_8UC3);
        this->back_fimage = cv::Mat(this->length.y+this->offset.y, this->offset.x, CV_8UC3);
        this->pimage = cv::Mat(this->length.y, this->length.x, CV_8UC3, cv::Scalar(this->color[0], this->color[1], this->color[2]));
        cv::Mat pimage_content = cv::Mat(this->pimage, cv::Rect(this->border.x, this->border.y, this->length.x - 2*this->border.x, this->length.y - 2*this->border.y) );
        cv::resize(this->src_pimage, pimage_content, pimage_content.size(), 0., 0., cv::INTER_CUBIC);
        //Compute orientations
        float s, c00, c01, c02, c03, c10, c11, c12, c13;
        this->fimage_mask[ORIENT::ORIENT_NW] = cv::Mat(this->offset.y + this->length.y, this->offset.x, CV_8UC1, cv::Scalar(0));
        s = +(float)(this->offset.y                  - 1)/(float)this->offset.x, c00 = 0., c01 = s, c02 = s / (float)this->offset.x, c03 = -s / (float)(this->offset.x * this->offset.x);
        s = +(float)(this->offset.y + this->length.y - 1)/(float)this->offset.x, c10 = 0., c11 = s, c12 = s / (float)this->offset.x, c13 = -s / (float)(this->offset.x * this->offset.x);
        for (unsigned int _i=0; _i!=this->offset.x; _i++) {
        	unsigned int y0 = this->length.y + this->offset.y - (unsigned int)std::round(c00 + c01*(float)(_i) + c02*(float)(_i*_i) + c03*(float)(_i*_i*_i));
            unsigned int y1 = this->length.y + this->offset.y - (unsigned int)std::round(c10 + c11*(float)(_i) + c12*(float)(_i*_i) + c13*(float)(_i*_i*_i));
        	do { unsigned char& c = this->fimage_mask[ORIENT::ORIENT_NW].at<unsigned char>(y0-1, _i); c = 1; } while(y1 != y0--);
            }
        this->frame[ORIENT::ORIENT_NW] = { 0, -(int)(this->length.y + this->offset.y), (int)(this->length.x + this->offset.x), 0 } ;
        this->fimage_mask[ORIENT::ORIENT_NE] = cv::Mat(this->offset.y + this->length.y, this->offset.x, CV_8UC1, cv::Scalar(0));
        s = -(float)(this->offset.y                  - 1)/(float)this->offset.x, c00 = (float)(this->offset.y),                  c01 = s, c02 = s / (float)this->offset.x, c03 = -s / (float)(this->offset.x * this->offset.x);
        s = -(float)(this->offset.y + this->length.y - 1)/(float)this->offset.x, c10 = (float)(this->offset.y + this->length.y), c11 = s, c12 = s / (float)this->offset.x, c13 = -s / (float)(this->offset.x * this->offset.x);
        for (unsigned int _i=0; _i!=this->offset.x; _i++) {
        	unsigned int y0 = this->length.y + this->offset.y - (unsigned int)std::round(c00 + c01*(float)(_i) + c02*(float)(_i*_i) + c03*(float)(_i*_i*_i));
            unsigned int y1 = this->length.y + this->offset.y - (unsigned int)std::round(c10 + c11*(float)(_i) + c12*(float)(_i*_i) + c13*(float)(_i*_i*_i));
        	do { unsigned char& c = this->fimage_mask[ORIENT::ORIENT_NE].at<unsigned char>(y0-1, _i); c = 1; } while(y1 != y0--);
            }
        this->frame[ORIENT::ORIENT_NE] = { -(int)(this->length.x + this->offset.x), -(int)(this->length.y + this->offset.y), 0, 0 } ;
        this->fimage_mask[ORIENT::ORIENT_SW] = cv::Mat(this->offset.y + this->length.y, this->offset.x, CV_8UC1, cv::Scalar(0));
        s = -(float)(this->offset.y + this->length.y - 1)/(float)this->offset.x, c00 = (float)(this->offset.y + this->length.y), c01 = s, c02 = s / (float)this->offset.x, c03 = -s / (float)(this->offset.x * this->offset.x);
        s = -(float)(this->offset.y                  - 1)/(float)this->offset.x, c10 = (float)(this->offset.y + this->length.y), c11 = s, c12 = s / (float)this->offset.x, c13 = -s / (float)(this->offset.x * this->offset.x);
        for (unsigned int _i=0; _i!=this->offset.x; _i++) {
        	unsigned int y0 = this->length.y + this->offset.y - (unsigned int)std::round(c00 + c01*(float)(_i) + c02*(float)(_i*_i) + c03*(float)(_i*_i*_i));
            unsigned int y1 = this->length.y + this->offset.y - (unsigned int)std::round(c10 + c11*(float)(_i) + c12*(float)(_i*_i) + c13*(float)(_i*_i*_i));
        	do { unsigned char& c = this->fimage_mask[ORIENT::ORIENT_SW].at<unsigned char>(y0-1, _i); c = 1; } while(y1 != y0--);
            }
        this->frame[ORIENT::ORIENT_SW] = { 0, 0, +(int)(this->length.x + this->offset.x), +(int)(this->length.y + this->offset.y) } ;
        this->fimage_mask[ORIENT::ORIENT_SE] = cv::Mat(this->offset.y + this->length.y, this->offset.x, CV_8UC1, cv::Scalar(0));
        s = +(float)(this->offset.y + this->length.y - 1)/(float)this->offset.x, c00 = 0.,                    c01 = s, c02 = s / (float)this->offset.x, c03 = -s / (float)(this->offset.x * this->offset.x);
        s = +(float)(this->offset.y                  - 1)/(float)this->offset.x, c10 = (float)this->length.y, c11 = s, c12 = s / (float)this->offset.x, c13 = -s / (float)(this->offset.x * this->offset.x);
        for (unsigned int _i=0; _i!=this->offset.x; _i++) {
        	unsigned int y0 = this->length.y + this->offset.y - (unsigned int)std::round(c00 + c01*(float)(_i) + c02*(float)(_i*_i) + c03*(float)(_i*_i*_i));
            unsigned int y1 = this->length.y + this->offset.y - (unsigned int)std::round(c10 + c11*(float)(_i) + c12*(float)(_i*_i) + c13*(float)(_i*_i*_i));
        	do { unsigned char& c = this->fimage_mask[ORIENT::ORIENT_SE].at<unsigned char>(y0-1, _i); c = 1; } while(y1 != y0--);
            }
        this->frame[ORIENT::ORIENT_SE] = { -(int)(this->length.x + this->offset.x), 0, 0, +(int)(this->length.y + this->offset.y) } ;
        //Init drawn
        this->_drawn_orient = (unsigned int)-1;
	}

    //This function changes orientation of the widget
	void set_default_orient(unsigned int default_orient) { if (default_orient < ORIENT::ORIENT_LAST) this->default_orient = default_orient; }

	//This function define container origin
	void set_origin(unsigned int x,unsigned int y) { this->origin.x = x, this->origin.y = y; }

	//This is the main function of the container
	void draw(cv::Mat& background_image) {
		unsigned int orient;
		if ( ((int)this->origin.x < background_image.cols) && ((int)(this->offset.x + this->length.x) <= background_image.cols) && ((int)this->origin.y < background_image.rows) && ((int)(this->offset.y + this->length.y) <= background_image.rows)) {
			if ((orient = this->forced_drawn_orient(background_image)) == (unsigned int)-1) orient = (unsigned int) this->default_orient;
			switch (orient){
				case ORIENT::ORIENT_NW : {
					cv::Mat pfragment(background_image,cv::Rect(this->origin.x + this->offset.x, this->origin.y - this->offset.y - this->length.y, this->length.x,                  this->length.y) );
					cv::Mat ffragment(background_image,cv::Rect(this->origin.x,                  this->origin.y - this->offset.y - this->length.y, this->offset.x, this->offset.y + this->length.y) );
					ffragment.copyTo(this->back_fimage), pfragment.copyTo(this->back_pimage); //Back-up the background
					ffragment.setTo(cv::Scalar(this->color[0], this->color[1], this->color[2]), this->fimage_mask[ORIENT::ORIENT_NW]), this->pimage.copyTo(pfragment); //Draw
				break; }
				case ORIENT::ORIENT_NE : {
					cv::Mat pfragment(background_image,cv::Rect(this->origin.x - this->offset.x - this->length.x, this->origin.y - this->offset.y - this->length.y, this->length.x,                  this->length.y) );
					cv::Mat ffragment(background_image,cv::Rect(this->origin.x - this->offset.x,                  this->origin.y - this->offset.y - this->length.y, this->offset.x, this->offset.y + this->length.y) );
					ffragment.copyTo(this->back_fimage), pfragment.copyTo(this->back_pimage); //Back-up the background
					ffragment.setTo(cv::Scalar(this->color[0], this->color[1], this->color[2]), this->fimage_mask[ORIENT::ORIENT_NE]), this->pimage.copyTo(pfragment); //Draw
				break; }
				case ORIENT::ORIENT_SW : {
					cv::Mat pfragment(background_image,cv::Rect(this->origin.x + this->offset.x, this->origin.y + this->offset.y, this->length.x,                  this->length.y) );
					cv::Mat ffragment(background_image,cv::Rect(this->origin.x,                  this->origin.y                 , this->offset.x, this->offset.y + this->length.y) );
					ffragment.copyTo(this->back_fimage), pfragment.copyTo(this->back_pimage); //Back-up the background
					ffragment.setTo(cv::Scalar(this->color[0], this->color[1], this->color[2]), this->fimage_mask[ORIENT::ORIENT_SW]), this->pimage.copyTo(pfragment); //Draw
				break; }
				case ORIENT::ORIENT_SE : {
					cv::Mat pfragment(background_image,cv::Rect(this->origin.x - this->offset.x - this->length.x, this->origin.y + this->offset.y, this->length.x,                  this->length.y) );
					cv::Mat ffragment(background_image,cv::Rect(this->origin.x - this->offset.x,                  this->origin.y                 , this->offset.x, this->offset.y + this->length.y) );
					ffragment.copyTo(this->back_fimage), pfragment.copyTo(this->back_pimage); //Back-up the background
					ffragment.setTo(cv::Scalar(this->color[0], this->color[1], this->color[2]), this->fimage_mask[ORIENT::ORIENT_SE]), this->pimage.copyTo(pfragment); //Draw
				break; }
				default                : throw("FAILED to draw unknown orientation of MapMarkerWidget");
			}
		this->_drawn_orient = orient;
		this->_drawn_frame.x = (int)this->origin.x + this->frame[orient].x, this->_drawn_frame.y = (int)this->origin.y + this->frame[orient].y, this->_drawn_frame.size_x = (int)this->origin.x + this->frame[orient].size_x, this->_drawn_frame.size_y = (int)this->origin.y + this->frame[orient].size_y;
		}
	}
	//This method reset the widget drawn flag to -1
	void reset_drawn_orient_flag(void) { this->_drawn_orient = (unsigned int)-1; }
    //This function undo the drawing
	void erase(cv::Mat& background_image) {
		if ( (this->_drawn_orient != (unsigned int)-1)) {
            switch (this->_drawn_orient){
				case ORIENT::ORIENT_NW : {
					cv::Mat ffragment(background_image,cv::Rect(this->origin.x,                  this->origin.y - this->offset.y - this->length.y, this->offset.x, this->offset.y + this->length.y) );
					cv::Mat pfragment(background_image,cv::Rect(this->origin.x + this->offset.x, this->origin.y - this->offset.y - this->length.y, this->length.x,                  this->length.y) );
					this->back_fimage.copyTo(ffragment), this->back_pimage.copyTo(pfragment);
				break; }
				case ORIENT::ORIENT_NE : {
					cv::Mat ffragment(background_image,cv::Rect(this->origin.x - this->offset.x,                  this->origin.y - this->offset.y - this->length.y, this->offset.x, this->offset.y + this->length.y) );
					cv::Mat pfragment(background_image,cv::Rect(this->origin.x - this->offset.x - this->length.x, this->origin.y - this->offset.y - this->length.y, this->length.x,                  this->length.y) );
					this->back_fimage.copyTo(ffragment), this->back_pimage.copyTo(pfragment);
				break; }
				case ORIENT::ORIENT_SW : {
					cv::Mat pfragment(background_image,cv::Rect(this->origin.x + this->offset.x, this->origin.y + this->offset.y, this->length.x,                  this->length.y) );
					cv::Mat ffragment(background_image,cv::Rect(this->origin.x,                  this->origin.y                 , this->offset.x, this->offset.y + this->length.y) );
					this->back_fimage.copyTo(ffragment), this->back_pimage.copyTo(pfragment);
				break; }
				case ORIENT::ORIENT_SE : {
					cv::Mat pfragment(background_image,cv::Rect(this->origin.x - this->offset.x - this->length.x, this->origin.y + this->offset.y, this->length.x,                  this->length.y) );
					cv::Mat ffragment(background_image,cv::Rect(this->origin.x - this->offset.x,                  this->origin.y                 , this->offset.x, this->offset.y + this->length.y) );
					this->back_fimage.copyTo(ffragment), this->back_pimage.copyTo(pfragment);
				break; }
				default                : throw("FAILED to erase unknown orientation of MapMarkerWidget");
			}
		  this->_drawn_orient = (unsigned int)-1;
		}
	}

};

//The map object
class MapWidget{
private :
	const unsigned char background_color[3] = {0xCC, 0xCC, 0xCC}; //Gray80
    coord2D_t<unsigned int> _window_size;    //Map window size
	coord2D_t<unsigned int> _window_center;  //Map center [0,1]
	rect2D_t<unsigned int>  _window_proj;    //Projective window
	cv::Mat     _image;  //The output image (final)

	std::vector<std::unique_ptr<MapMarkerWidget>> MapMarkerWidgets;
	std::vector<coord2D_t<unsigned int>> marker_coords2D;
	std::vector<unsigned int> drawn_markers, tmp_markers; //A stack actually but with random read access
public :
	cv::Mat background_image; //The background map image
    const coord2D_t<unsigned int>& window_size;    //Read-only map window size
	const coord2D_t<unsigned int>& window_center;  //Read-only map center
	const rect2D_t<unsigned int>&  window_proj;    //Read-only map window projection
	const cv::Mat& image;     //The read-only output map image


	//The constructor
	MapWidget(unsigned int window_center_x, unsigned int window_center_y, unsigned int window_size_x, unsigned int window_size_y, const cv::Mat& background_image) :
		window_size(this->_window_size),
		window_center(this->_window_center),
		window_proj(this->_window_proj),
		image(this->_image) {
    	if ( ((int)window_center_x >= background_image.cols) || ((int)window_center_y >= background_image.rows) ) throw("Incorrect map center requested");
    	else { this->_window_center.x = window_center_x, this->_window_center.y = window_center_y; }
    	if ( ( (window_size_x%2))||( (window_size_y%2)) ) throw("Odd output map size requested");
	    this->_window_size.x = window_size_x, this->_window_size.y = window_size_y, this->background_image = background_image;
        this->_image = cv::Mat(this->_window_size.y, this->_window_size.x,  CV_8UC3);
	    this->draw();
	}

	//This routine assign a new center. It retuns +1 if the draw function need to be called
	char set_center(unsigned int window_center_x, unsigned int window_center_y) {
		if ( ((int)window_center_x >= this->background_image.cols) && ((int)window_center_y >= this->background_image.rows) ) return -1;
		if ( (this->_window_center.x == window_center_x) && (this->_window_center.y == window_center_y) ) return 0;
		else {
			if ((int)window_center_x < this->background_image.cols) this->_window_center.x = window_center_x;
			if ((int)window_center_y < this->background_image.rows) this->_window_center.y = window_center_y;
			return +1;
		}
	}

	//The full map re-drawing
    void draw(void) {
        this->_window_proj.x = (this->_window_center.x < this->_window_size.x/2) ? this->_window_center.x : this->_window_size.x/2, this->_window_proj.size_x = (this->background_image.cols - this->_window_center.x < this->_window_size.x/2 ) ? this->background_image.cols - this->_window_center.x : this->_window_size.x/2;
        this->_window_proj.y = (this->_window_center.y < this->_window_size.y/2) ? this->_window_center.y : this->_window_size.y/2, this->_window_proj.size_y = (this->background_image.rows - this->_window_center.y < this->_window_size.y/2 ) ? this->background_image.rows - this->_window_center.y : this->_window_size.y/2;
    	//Draw a background
        this->_image.setTo(cv::Scalar(MapWidget::background_color[0], MapWidget::background_color[1], MapWidget::background_color[2]));
        cv::Mat fragment_src(this->background_image, cv::Rect(this->_window_center.x - this->_window_proj.x, this->_window_center.y - this->_window_proj.y, this->_window_proj.x + this->_window_proj.size_x, this->_window_proj.y + this->_window_proj.size_y));
        cv::Mat fragment_dst(this->_image,           cv::Rect(this->_window_size.x/2 - this->_window_proj.x, this->_window_size.y/2 - this->_window_proj.y, this->_window_proj.x + this->_window_proj.size_x, this->_window_proj.y + this->_window_proj.size_y));
        fragment_src.copyTo(fragment_dst);
        //Draw markers
        this->drawn_markers.clear();
        for ( unsigned int _i=0 ; _i != this->MapMarkerWidgets.size(); _i++ )
        	if ( (this->marker_coords2D[_i].x >= this->_window_center.x - this->_window_proj.x) && (this->marker_coords2D[_i].x < this->_window_center.x + this->_window_proj.size_x) && (this->marker_coords2D[_i].y >= this->_window_center.y - this->_window_proj.y) && (this->marker_coords2D[_i].y < this->_window_center.y + this->_window_proj.size_y) ) {
        		this->MapMarkerWidgets[_i]->set_origin(this->marker_coords2D[_i].x - this->_window_center.x + this->_window_size.x/2, this->marker_coords2D[_i].y - this->_window_center.y + this->_window_size.y/2);
        		this->MapMarkerWidgets[_i]->draw(this->_image);
        		this->drawn_markers.push_back(_i);
            }
        	else this->MapMarkerWidgets[_i]->reset_drawn_orient_flag();
    }

    //This method register photo widget in the map
    unsigned int marker_add(unsigned int map_x,unsigned int map_y,unsigned char color[3], float fshape_x, float fshape_y, unsigned int border_x, unsigned int border_y, unsigned int length_x, unsigned int length_y, const cv::Mat& src_pimage) {
    	unsigned int marker_id = (unsigned int)this->MapMarkerWidgets.size();
    	this->MapMarkerWidgets.push_back(std::make_unique<MapMarkerWidget>(color, fshape_x, fshape_y, border_x, border_y, length_x, length_y, src_pimage));
  		this->MapMarkerWidgets.back()->set_origin((unsigned int)-1, (unsigned int)-1);
  		this->marker_coords2D.push_back({ .x = map_x, .y = map_y });
  		return marker_id;
    }

    //The function update marker on the map
    void marker_update(unsigned int marker_id, unsigned int x, unsigned int y) {
    	if ( (marker_id < this->MapMarkerWidgets.size()) && ( (this->marker_coords2D[marker_id].x != x) || (this->marker_coords2D[marker_id].y != y) ) ) {
    		if (this->MapMarkerWidgets[marker_id]->drawn_orient != (unsigned int)-1) { //Erase marker(s) first using bread-first method to check for overlaps
    			this->tmp_markers.push_back(marker_id);
    			unsigned int _k = this->drawn_markers.size(); do _k--; while (this->drawn_markers[_k] != marker_id);
    			unsigned int _i = _k;
    			while (++_i != this->drawn_markers.size())
    				for (unsigned int _j=0 ; _j != this->tmp_markers.size(); _j++)
    					if ( (rect_overlap(this->MapMarkerWidgets[this->drawn_markers[_i]]->drawn_frame, this->MapMarkerWidgets[this->tmp_markers[_j]]->drawn_frame))) {
        					this->tmp_markers.push_back(this->drawn_markers[_i]);
        					break;
        				}
    			_i = this->tmp_markers.size(); while ( (_i--)) this->MapMarkerWidgets[this->tmp_markers[_i]]->erase(this->_image);
                this->tmp_markers.clear();
    			while (++_k != this->drawn_markers.size()) {
    				if (this->MapMarkerWidgets[this->drawn_markers[_k]]->drawn_orient == (unsigned int)-1 )
    					this->MapMarkerWidgets[this->drawn_markers[_k]]->draw(this->_image);
    				this->drawn_markers[_k-1] = this->drawn_markers[_k];
    			}
                this->drawn_markers[_k-1] = marker_id;     //Put the last moved into the end of stack as it is expected to move more
    		}
    		else this->drawn_markers.push_back(marker_id); //Put the last moved into the end of stack as it is expected to move more
    		//Draw the marker
    		this->marker_coords2D[marker_id] = { x, y };
    		if ( (x >= this->_window_center.x - this->_window_proj.x) && (x < this->_window_center.x + this->_window_proj.size_x) && (y >= this->_window_center.y - this->_window_proj.y) && (y < this->_window_center.y + this->_window_proj.size_y) ) {
    			this->MapMarkerWidgets[marker_id]->set_origin(x - this->_window_center.x + this->_window_size.x/2, y - this->_window_center.y + this->_window_size.y/2);
    			this->MapMarkerWidgets[marker_id]->draw(this->_image);
    		}
    	}
    }

};




#define BACKGROUND_IMAGE "/home/alex/eclipse/opencv-work/opencv_map_widget/downtown.jpeg"
#define PLAYER_IMAGE "/home/alex/eclipse/opencv-work/opencv_map_widget/happy_fish.jpg"

int main( int argc, char** argv )
{
cv::Mat background_image, player_image, image;

player_image = cv::imread(PLAYER_IMAGE, CV_LOAD_IMAGE_COLOR );
background_image = cv::imread(BACKGROUND_IMAGE, CV_LOAD_IMAGE_COLOR );

cv::namedWindow( "OutputWindow", CV_WINDOW_AUTOSIZE );

//coord2D_t<unsigned int> map_origin = { .x = 4644, .y = 2422 };
//coord2D_t<unsigned int> map_origin = { .x = 0, .y = 0 };
coord2D_t<unsigned int> map_origin = { .x = 5335, .y = 3263 };
unsigned char color[3];

//Create map widget
MapWidget map = MapWidget(map_origin.x, map_origin.y, 640, 480, background_image);
color[0] = 255, color[1] =   0, color[2] =   0; map.marker_add(map_origin.x, map_origin.y, color, 0.2, 0.4, 4, 4, 64, 64, player_image);
color[0] =   0, color[1] = 255, color[2] =   0; map.marker_add(map_origin.x, map_origin.y, color, 0.2, 0.4, 4, 4, 64, 64, player_image);
color[0] =   0, color[1] =   0, color[2] = 255; map.marker_add(map_origin.x, map_origin.y, color, 0.2, 0.4, 4, 4, 64, 64, player_image);
color[0] = 127, color[1] = 127, color[2] = 127; map.marker_add(map_origin.x, map_origin.y, color, 0.2, 0.4, 4, 4, 64, 64, player_image);
map.draw();
cv::imshow( "OutputWindow", map.image);
cv::waitKey(10);

coord2D_t<unsigned int> markers_coord2D[4] = { { .x = map_origin.x, .y = map_origin.y }, { .x = map_origin.x, .y = map_origin.y }, { .x = map_origin.x, .y = map_origin.y }, { .x = map_origin.x, .y = map_origin.y } };
std::srand(2004);
for (unsigned int _i=0 ; _i!=10000; _i++) {
	//Map movement
	if (!(_i%10)) {
		bool flag = false;
		switch (std::rand() % 3) {
			case  1 : { if (map_origin.x < map.background_image.cols - map.window_size.x/2 - 10) { map_origin.x+=10, flag = true; } break; }
			case  2 : { if (map_origin.x > map.window_size.x/2 + 10)                             { map_origin.x-=10, flag = true; } break; }
			default : ;
		}
		switch (std::rand() % 3) {
			case  1 : { if (map_origin.y < map.background_image.rows - map.window_size.y/2 - 10) { map_origin.y+=10, flag = true; } break; }
			case  2 : { if (map_origin.y > map.window_size.y/2 + 10)                             { map_origin.y-=10, flag = true; } break; }
			default : ;
		}
		if ( (flag))
			if (map.set_center(map_origin.x, map_origin.y) == +1)
				map.draw();
	}
	else {
		//Marker movements
		for (unsigned int _j=0 ; _j!=4; _j++) {
			bool flag = false;
			switch (std::rand() % 3) {
				case  1 : { if (markers_coord2D[_j].x <  map_origin.x + map.window_proj.size_x - 10) { markers_coord2D[_j].x+=10, flag = true; } break; }
				case  2 : { if (markers_coord2D[_j].x >= map_origin.x - map.window_proj.x      + 10) { markers_coord2D[_j].x-=10, flag = true; } break; }
				default : ;
			}
			switch (std::rand() % 3) {
				case  1 : { if (markers_coord2D[_j].y <  map_origin.y + map.window_proj.size_y - 10) { markers_coord2D[_j].y+=10, flag = true; } break; }
				case  2 : { if (markers_coord2D[_j].y >= map_origin.y - map.window_proj.y      + 10) { markers_coord2D[_j].y-=10, flag = true; } break; }
				default : ;
			}
			if ( (flag)) map.marker_update(_j, markers_coord2D[_j].x, markers_coord2D[_j].y);
		}
	}
    cv::imshow( "OutputWindow", map.image);
	cv::waitKey(10);

	cout << _i << "-th step\n" << endl;
}

cv::imshow( "OutputWindow", map.image);
cv::waitKey(0);


return 0;
}




