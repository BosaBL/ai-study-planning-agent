import axios from "axios";

const API_BASE = "http://131.221.33.104";

export const uploadPDFs = async (files) => {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  return await axios.post(`${API_BASE}/upload-pdfs/`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

export const generateGuide = async (data) => {
  const response = await axios.post(`${API_BASE}/generate-study-guide-async/`, data);
  return response;
};

export const getGuideById = (guideId) => {
  return axios.get(`${API_BASE}/api/summaries/${guideId}`);
};
