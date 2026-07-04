import React from "react";
import ReactDOM from "react-dom/client";
import { ProductionMixDemo } from "./ProductionMixDemo";
import "./index.css";

const API_URL =
  import.meta.env.VITE_API_URL || "https://linprogx-demo.onrender.com";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ProductionMixDemo apiUrl={API_URL} />
  </React.StrictMode>
);
