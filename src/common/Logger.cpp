/*
Copyright (c) 2012-2020 Maarten Baert <maarten-baert@hotmail.com>

This file is part of SimpleScreenRecorder.

SimpleScreenRecorder is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SimpleScreenRecorder is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with SimpleScreenRecorder.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "Logger.h"

#include "QueueBuffer.h"

Logger *Logger::s_instance = NULL;

static QString LogFormatTime() {
	return QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss.zzz");
}

Logger::Logger() {
	assert(s_instance == NULL);
	qRegisterMetaType<enum_type>();
	m_capture_pipe[0] = -1;
	m_capture_pipe[1] = -1;
	m_original_stderr = -1;
	try {
		Init();
	} catch(...) {
		Free();
		throw;
	}
	s_instance = this;
}

Logger::~Logger() {
	assert(s_instance == this);
	s_instance = NULL;
	Free();
}

void Logger::SetLogFile(const QString &filename) {
	m_log_file.setFileName(filename);
	m_log_file.open(QFile::WriteOnly | QFile::Append | QFile::Text | QFile::Unbuffered);
}

void Logger::LogDrop(const QString& str) {
	//printf("str: %s\n", str.toStdString().c_str());
	std::string stdstr = str.toStdString();
	enum_type drop_type = TYPE_INFO;
	QString qstr_no_color = str;
	//printf("stdstr.length(): %d\n", stdstr.length());
	if(stdstr.length() >= 9 && stdstr.compare(stdstr.length() - 4, 4, "\033[0m") == 0)
	{
		//printf("in if\n");
		if(stdstr.compare(0, 5, "\033[91m") == 0)
			drop_type = TYPE_INFO_GOLD;
		else if(stdstr.compare(0, 5, "\033[94m") == 0)
			drop_type = TYPE_INFO_BLUE;
		else if(stdstr.compare(0, 5, "\033[95m") == 0)
			drop_type = TYPE_INFO_ORANGE;
		else if(stdstr.compare(0, 5, "\033[92m") == 0)
			drop_type = TYPE_INFO_GREEN;
		else if(stdstr.compare(0, 5, "\033[93m") == 0)
			drop_type = TYPE_INFO_YELLOW;
		else if(stdstr.compare(0, 5, "\033[39m") == 0)
			drop_type = TYPE_INFO_WHITE;
		else if(stdstr.compare(0, 5, "\033[90m") == 0)
			drop_type = TYPE_INFO_GRAY;
		else
		  assert(0);
		qstr_no_color = stdstr.substr(5, stdstr.length() - 5 - 4).c_str();
	}
	printf("qstr_no_color: %s\n", qstr_no_color.toStdString().c_str());
	assert(s_instance != NULL);
	std::lock_guard<std::mutex> lock(s_instance->m_mutex); Q_UNUSED(lock);
	QByteArray buf = (str + "\n").toLocal8Bit();
	write(s_instance->m_original_stderr, buf.constData(), buf.size());
	if(s_instance->m_log_file.isOpen())
		s_instance->m_log_file.write((LogFormatTime() + " (I) " + str + "\n").toLocal8Bit());
	emit s_instance->NewLine(drop_type, qstr_no_color);
}

void Logger::LogInfo(const QString& str) {
	assert(s_instance != NULL);
	std::lock_guard<std::mutex> lock(s_instance->m_mutex); Q_UNUSED(lock);
	QByteArray buf = (str + "\n").toLocal8Bit();
	write(s_instance->m_original_stderr, buf.constData(), buf.size());
	if(s_instance->m_log_file.isOpen())
		s_instance->m_log_file.write((LogFormatTime() + " (I) " + str + "\n").toLocal8Bit());
	emit s_instance->NewLine(TYPE_INFO, str);
}

void Logger::LogWarning(const QString& str) {
	assert(s_instance != NULL);
	std::lock_guard<std::mutex> lock(s_instance->m_mutex); Q_UNUSED(lock);
	QByteArray buf = ("\033[1;33m" + str + "\033[0m\n").toLocal8Bit();
	write(s_instance->m_original_stderr, buf.constData(), buf.size());
	if(s_instance->m_log_file.isOpen())
		s_instance->m_log_file.write((LogFormatTime() + " (W) " + str + "\n").toLocal8Bit());
	emit s_instance->NewLine(TYPE_WARNING, str);
}

void Logger::LogError(const QString& str) {
	assert(s_instance != NULL);
	std::lock_guard<std::mutex> lock(s_instance->m_mutex); Q_UNUSED(lock);
	QByteArray buf = ("\033[1;31m" + str + "\033[0m\n").toLocal8Bit();
	write(s_instance->m_original_stderr, buf.constData(), buf.size());
	if(s_instance->m_log_file.isOpen())
		s_instance->m_log_file.write((LogFormatTime() + " (E) " + str + "\n").toLocal8Bit());
	emit s_instance->NewLine(TYPE_ERROR, str);
}

void Logger::Init() {
	if(pipe2(m_capture_pipe, O_CLOEXEC) != 0)
		throw std::runtime_error("Failed to create capture pipe");
	m_original_stderr = dup(2); // copy stderr
	dup2(m_capture_pipe[1], 2); // redirect stderr
	m_capture_thread = std::thread(&Logger::CaptureThread, this);
}

void Logger::Free() {
	if(m_original_stderr != -1) {
		dup2(m_original_stderr, 2); // restore stderr
	}
	if(m_capture_pipe[1] != -1) {
		close(m_capture_pipe[1]); // close write end of pipe
		m_capture_pipe[1] = -1;
	}
	if(m_capture_thread.joinable()) {
		m_capture_thread.join(); // wait for thread
	}
	if(m_capture_pipe[0] != -1) {
		close(m_capture_pipe[0]); // close read end of pipe
		m_capture_pipe[0] = -1;
	}
	if(m_original_stderr != -1) {
		close(m_original_stderr); // close copy of stderr
		m_original_stderr = -1;
	}
}

void Logger::CaptureThread() {
	QueueBuffer<char> buffer;
	size_t pos = 0;
	for( ; ; ) {
		ssize_t num;
		do {
			num = read(m_capture_pipe[0], buffer.Reserve(PIPE_BUF), PIPE_BUF);
		} while(num == -1 && errno == EINTR);
		if(num <= 0)
			break;
		buffer.Push(num);
		while(pos < buffer.GetSize()) {
			if(buffer[pos] == '\n') {
				std::lock_guard<std::mutex> lock(s_instance->m_mutex); Q_UNUSED(lock);
				std::string buf = "\033[2m" + std::string(buffer.GetData(), pos) + "\033[0m\n";
				write(s_instance->m_original_stderr, buf.data(), buf.size());
				QString str = QString::fromLocal8Bit(buffer.GetData(), pos);
				if(s_instance->m_log_file.isOpen())
					s_instance->m_log_file.write((LogFormatTime() + " (S) " + str + "\n").toLocal8Bit());
				emit s_instance->NewLine(TYPE_STDERR, str);
				buffer.Pop(pos + 1);
				pos = 0;
			} else {
				++pos;
			}
		}
	}
}
