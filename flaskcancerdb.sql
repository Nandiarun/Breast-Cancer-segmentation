-- phpMyAdmin SQL Dump
-- version 4.8.2
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: May 21, 2021 at 06:25 PM
-- Server version: 10.1.34-MariaDB
-- PHP Version: 7.2.7

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `flaskplantleafdb`
--

-- --------------------------------------------------------

--
-- Table structure for table `userdata`
--

CREATE TABLE `userdata` (
  `Named` varchar(50) DEFAULT NULL,
  `Email` varchar(50) DEFAULT NULL,
  `Pswd` varchar(50) DEFAULT NULL,
  `Phone` varchar(50) DEFAULT NULL,
  `Addr` varchar(4000) DEFAULT NULL,
  `Dob` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `userdata`
--

INSERT INTO `userdata` (`Named`, `Email`, `Pswd`, `Phone`, `Addr`, `Dob`) VALUES
('Sunny Boyka', 'madhsunil@gmail.com', 'qqq', '9036453696', 'Mysore\njj', '12-12-1978'),
('Sunny Boyka', 'madhsunil@gmail.com', 'qaz', '9036453696', 'Mysore\njj', '05/01/2021'),
('Sunny Boyka', 'madhsunil@gmail.com', 'q', '9036453696', 'Mysore\njj', '05/01/2021'),
('Sunny Boyka', 'madhsunil@gmail.com', 'q', '9036453696', 'Mysore\njj', '05/01/2021'),
('Sunny Boyka', 'madhsunil@gmail.com', 'qaz', '9036453696', 'Mysore\njj', '05/01/2021'),
('Vinay Kumar', 'vinaykumarkn66@gmail.com', 'qazwsx', '7894561230', 'Mandya', '11/26/2020');
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
