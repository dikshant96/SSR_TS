import numpy as np
from scipy import interpolate
import os


class OFPy:
    def __init__(self,dir,files=0,mesh=0,time=0):
        self.case_dir = dir
        self.last_time, self.dir = self.get_timestep(time=0)
        self.files = files
        self.mesh = mesh
        self.fnames = self.get_filenames()
        self.get_grid()

    def get_timestep(self,time=0):
        if time == 0:
            files = os.listdir(self.case_dir)

        time_files = []
        time_files_int = []
        for file in files:
            try:
                int(file)
                time_files.append(file)
                time_files_int.append(int(file))
            except ValueError:
                pass

        time_files_int = np.array(time_files_int)
        ind = np.argmax(time_files_int)

        tstep = time_files_int[ind]
        dir_tstep = self.case_dir + time_files[ind] + '/'
        return tstep, dir_tstep

    def get_grid(self):
        C_vars = self.read_internalvector(self.fnames[0])
        self.x = C_vars[:,0]
        self.y = C_vars[:,1]
        self.z = C_vars[:,2]

    def get_filenames(self):
        fnames = []
        for i, file in enumerate(self.files):
            fname = self.dir + self.files[file]
            fnames.append(fname)
        return fnames

    def read_internalvector(self,fname):
        """
        :param fname: filename for the parameter
        :return: vector of field values at cell centres
        """
        f = open(fname,'r')
        lines = f.readlines()
        f.close()
        start = 1e6
        for i, line in enumerate(lines):
            if 'internalField' in line:
                start = i + 2
                entries_len = int(lines[i + 1])
                vals = np.zeros((entries_len, 3))
                j = 0
            elif start < i <= entries_len + start:
                line = line.replace('(', '')
                line = line.replace(')', '')
                line = line.split(' ')
                vals[j, 0] = line[0]
                vals[j, 1] = line[1]
                vals[j, 2] = line[2]
                j = j + 1
        return vals

    def read_internalsymmtensor(self,fname):
        """
        :param fname: filename for the parameter
        :return: vector of field values at cell centres
        """
        f = open(fname,'r')
        lines = f.readlines()
        f.close()
        start = 1e6
        for i, line in enumerate(lines):
            if 'internalField' in line:
                start = i + 2
                entries_len = int(lines[i + 1])
                vals = np.zeros((entries_len, 6))
                j = 0
            elif start < i <= entries_len + start:
                line = line.replace('(', '')
                line = line.replace(')', '')
                line = line.split(' ')
                vals[j, 0] = line[0]
                vals[j, 1] = line[1]
                vals[j, 2] = line[2]
                vals[j, 3] = line[3]
                vals[j, 4] = line[4]
                vals[j, 5] = line[5]
                j = j + 1
        return vals

    def read_internaltensor(self,fname):
        """
        :param fname: filename for the parameter
        :return: vector of field values at cell centres
        """
        f = open(fname,'r')
        lines = f.readlines()
        f.close()
        start = 1e6
        for i, line in enumerate(lines):
            if 'internalField' in line:
                start = i + 2
                entries_len = int(lines[i + 1])
                vals = np.zeros((entries_len, 9))
                j = 0
            elif start < i <= entries_len + start:
                line = line.replace('(', '')
                line = line.replace(')', '')
                line = line.split(' ')
                vals[j, 0] = line[0]
                vals[j, 1] = line[1]
                vals[j, 2] = line[2]
                vals[j, 3] = line[3]
                vals[j, 4] = line[4]
                vals[j, 5] = line[5]
                vals[j, 6] = line[6]
                vals[j, 7] = line[7]
                vals[j, 8] = line[8]
                j = j + 1
        return vals


    def read_internalscalar(self,fname):
        """
        :param fname: filename for the parameter
        :return: scalar of field values at cell centres
        """
        f = open(fname,'r')
        lines = f.readlines()
        f.close()
        start = 1e6
        for i, line in enumerate(lines):
            if 'internalField' in line:
                start = i + 2
                entries_len = int(lines[i + 1])
                vals = np.zeros((entries_len, 1))
                j = 0
            elif start < i <= entries_len + start:
                line = line.replace('(', '')
                line = line.replace(')', '')
                vals[j,0] = float(line[:-1])
                j = j + 1
        return vals

    def read_boundaryvector(self,fname,wallname):
        """
        :param fname: filename for the parameter
        :return: vector of field values at cell centres
        """
        f = open(fname,'r')
        lines = f.readlines()
        f.close()
        start = 1e7
        entries_len = 1e8
        for i, line in enumerate(lines):
            if wallname in line:
                start = i + 5
                entries_len = int(lines[i + 4])
                vals = np.zeros((entries_len, 3))
                j = 0
            elif start < i <= entries_len + start:
                line = line.replace('(', '')
                line = line.replace(')', '')
                line = line.split(' ')
                vals[j, 0] = line[0]
                vals[j, 1] = line[1]
                vals[j, 2] = line[2]
                j = j + 1
        return vals

    def read_boundarytensor(self,fname,wallname):
        """
        :param fname: filename for the parameter
        :wallname : wall name of boundary in OF
        :return: vector of field values at cell centres
        """
        f = open(fname,'r')
        lines = f.readlines()
        f.close()
        start = 1e7
        entries_len = 1e8
        for i, line in enumerate(lines):
            if wallname in line:
                start = i + 5
                entries_len = int(lines[i + 4])
                vals = np.zeros((entries_len, 9))
                j = 0
            elif start < i <= entries_len + start:
                line = line.replace('(', '')
                line = line.replace(')', '')
                line = line.split(' ')
                vals[j, 0] = line[0]
                vals[j, 1] = line[1]
                vals[j, 2] = line[2]
                vals[j, 3] = line[3]
                vals[j, 4] = line[4]
                vals[j, 5] = line[5]
                vals[j, 6] = line[6]
                vals[j, 7] = line[7]
                vals[j, 8] = line[8]
                j = j + 1
        return vals
