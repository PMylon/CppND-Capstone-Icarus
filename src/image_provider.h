#ifndef IMAGE_PROVIDER_H_
#define IMAGE_PROVIDER_H_

#include <opencv2/opencv.hpp>
#include <cstdint>
#include <array>
#include <string>
#include <vector>
#include <filesystem>

enum class ColorFormat : uint8_t
{
    RGB = 0,
    BGR = 1
};

enum class MemoryLayout : uint8_t
{
    HWC = 0,
    CHW = 1
};

struct Image
{
    cv::Mat matrix;
    int height;
    int  width;
    ColorFormat fmt;
    MemoryLayout layout;
    std::string path;
};

class ImageIterator
{
    public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::filesystem::path;
    using reference = const value_type&;
    using pointer = const value_type*;
    using difference_type = std::ptrdiff_t;

    ImageIterator(std::vector<value_type>::const_iterator imageEntryIt
    , std::vector<value_type>::const_iterator imageEntryItBegin
    , std::vector<value_type>::const_iterator imageEntryItEnd) : 
    imageEntryIt_{imageEntryIt},  imageEntryItBegin_{imageEntryItBegin}, imageEntryItEnd_{imageEntryItEnd}{}
    reference operator*() const { return *imageEntryIt_; }
    pointer operator->() const { return &(*imageEntryIt_); }
    ImageIterator& operator++()
    {
        ++imageEntryIt_;
        if (imageEntryIt_ == imageEntryItEnd_)
        {
            imageEntryIt_ = imageEntryItBegin_;
        }

        return *this;
    }
    ImageIterator operator++(int) { ImageIterator tempIt{*this}; operator++(); return tempIt; }

    friend bool operator==(const ImageIterator& it1, const ImageIterator& it2) { return (it1.imageEntryIt_ == it2.imageEntryIt_); }
    friend bool operator!=(const ImageIterator& it1, const ImageIterator& it2) { return !(it1.imageEntryIt_ == it2.imageEntryIt_); }

    private:
    std::vector<value_type>::const_iterator imageEntryIt_;
    std::vector<value_type>::const_iterator imageEntryItBegin_;
    std::vector<value_type>::const_iterator imageEntryItEnd_;
};

class ImageProvider
{
    public:
    ImageProvider() : imageEntries_{populateImageEntries()}, imageIt_{imageEntries_.cbegin(), imageEntries_.cbegin(), imageEntries_.cend()} {}
    Image GetImage();

    private:
    static std::vector<std::filesystem::path> populateImageEntries()
    {
        std::vector<std::filesystem::path> imageEntries;

        for (const auto& imageEntry : std::filesystem::directory_iterator{imagesPath_})
        {
            imageEntries.push_back(imageEntry.path());
        }

        return imageEntries;
    }

    static constexpr const char* const imagesPath_ = "assets/images/";
    std::vector<std::filesystem::path> imageEntries_;
    ImageIterator imageIt_;
};

#endif // #ifndef IMAGE_PROVIDER_H_