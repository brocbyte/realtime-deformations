#pragma once
#include <iostream>
#include <glm/glm.hpp>
#include <constants.hpp>

class Logger {
public:
    enum class LogLevel {
        INFO,
        WARNING,
        ERROR
    };
    void log(LogLevel level, const std::string& iMessage) {
        if (level >= _level) {
            std::cout << iMessage << "\n";
        }
    };
    void log(LogLevel level, const std::string& iName, ftype iValue) {
        log(level, iName + " is: " + std::to_string(iValue));
    }
    void log(LogLevel level, const std::string& iName, int iValue) {
        log(level, iName + " is: " + std::to_string(iValue));
    }
    void log(LogLevel level, const std::string& iName, const glm::vec3& iVec) {
        //log(level, iName + " is: " + glm::to_string(iVec));
    }
    void log(LogLevel level, const std::string& iName, const glm::mat3& iMat) {
        //log(level, iName + " is: " + glm::to_string(iMat));
    }
    void setLevel(LogLevel level) {
        _level = level;
    }
private:
    LogLevel _level = LogLevel::INFO;
};


class Loggable {
public:
    void setLevel(Logger::LogLevel level) {
        logger.setLevel(level);
    }
protected:
    Logger logger;
};
