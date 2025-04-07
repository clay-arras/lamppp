#ifndef DUMMY_VALUE_H
#define DUMMY_VALUE_H

#include <memory>

class Float {
public:
    Float();
    explicit Float(float value);
    
    Float(const Float& other);
    Float& operator=(const Float& other);
    
    Float(Float&& other) noexcept;
    Float& operator=(Float&& other) noexcept;
    
    float getValue() const;
    void setValue(float value);
    
    Float operator+(const Float& other) const;
    Float operator-(const Float& other) const;
    Float operator*(const Float& other) const;
    Float operator/(const Float& other) const;
    
    Float& operator+=(const Float& other);
    Float& operator-=(const Float& other);
    Float& operator*=(const Float& other);
    Float& operator/=(const Float& other);
    
    bool operator==(const Float& other) const;
    bool operator!=(const Float& other) const;
    bool operator<(const Float& other) const;
    bool operator<=(const Float& other) const;
    bool operator>(const Float& other) const;
    bool operator>=(const Float& other) const;

private:
    struct Impl {
        explicit Impl(float val = 0.0F) : value(val) {}
        float value;
    };
    
    std::unique_ptr<Impl> impl_;
};

#endif // DUMMY_VALUE_H
